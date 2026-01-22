import hashlib
from hashlib import sha1
import importlib
from itertools import chain
import json
import logging
import os
from os.path import isfile
from os.path import join
from re import compile as regex_compile
import sys
from warnings import warn as _warn
import requests
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import md
from saml2 import saml
from saml2 import samlp
from saml2 import xmldsig
from saml2 import xmlenc
from saml2.extension.algsupport import NAMESPACE as NS_ALGSUPPORT
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.extension.idpdisc import BINDING_DISCO
from saml2.extension.idpdisc import DiscoveryResponse
from saml2.extension.mdattr import NAMESPACE as NS_MDATTR
from saml2.extension.mdattr import EntityAttributes
from saml2.extension.mdrpi import NAMESPACE as NS_MDRPI
from saml2.extension.mdrpi import RegistrationInfo
from saml2.extension.mdrpi import RegistrationPolicy
from saml2.extension.mdui import NAMESPACE as NS_MDUI
from saml2.extension.mdui import Description
from saml2.extension.mdui import DisplayName
from saml2.extension.mdui import InformationURL
from saml2.extension.mdui import Logo
from saml2.extension.mdui import PrivacyStatementURL
from saml2.extension.mdui import UIInfo
from saml2.extension.shibmd import NAMESPACE as NS_SHIBMD
from saml2.extension.shibmd import Scope
from saml2.httpbase import HTTPBase
from saml2.md import NAMESPACE as NS_MD
from saml2.md import ArtifactResolutionService
from saml2.md import EntitiesDescriptor
from saml2.md import EntityDescriptor
from saml2.md import NameIDMappingService
from saml2.md import SingleSignOnService
from saml2.mdie import to_dict
from saml2.s_utils import UnknownSystemEntity
from saml2.s_utils import UnsupportedBinding
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import split_len
from saml2.time_util import add_duration
from saml2.time_util import before
from saml2.time_util import instant
from saml2.time_util import str_to_time
from saml2.time_util import valid
from saml2.validate import NotValid
from saml2.validate import valid_instance
class InMemoryMetaData(MetaData):

    def __init__(self, attrc, metadata='', node_name=None, check_validity=True, security=None, **kwargs):
        super().__init__(attrc, metadata=metadata)
        self.entity = {}
        self.security = security
        self.node_name = node_name
        self.entities_descr = None
        self.entity_descr = None
        self.check_validity = check_validity
        try:
            self.filter = kwargs['filter']
        except KeyError:
            self.filter = None

    def items(self):
        return self.entity.items()

    def keys(self):
        return self.entity.keys()

    def values(self):
        return self.entity.values()

    def __len__(self):
        return len(self.entity)

    def __contains__(self, item):
        return item in self.entity.keys()

    def __getitem__(self, item):
        return self.entity[item]

    def __setitem__(self, key, value):
        self.entity[key] = value

    def __delitem__(self, key):
        del self.entity[key]

    def do_entity_descriptor(self, entity_descr):
        if self.check_validity:
            try:
                if not valid(entity_descr.valid_until):
                    logger.error('Entity descriptor (entity id:%s) too old', entity_descr.entity_id)
                    self.to_old.append(entity_descr.entity_id)
                    return
            except AttributeError:
                pass
        if entity_descr.entity_id in self.entity:
            print(f"Duplicated Entity descriptor (entity id: '{entity_descr.entity_id}')", file=sys.stderr)
            return
        _ent = to_dict(entity_descr, metadata_modules())
        flag = 0
        for descr in ['spsso', 'idpsso', 'role', 'authn_authority', 'attribute_authority', 'pdp', 'affiliation']:
            _res = []
            try:
                _items = _ent[f'{descr}_descriptor']
            except KeyError:
                continue
            if descr == 'affiliation':
                flag += 1
                continue
            for item in _items:
                for prot in item['protocol_support_enumeration'].split(' '):
                    if prot == samlp.NAMESPACE:
                        item['protocol_support_enumeration'] = prot
                        _res.append(item)
                        break
            if not _res:
                del _ent[f'{descr}_descriptor']
            else:
                flag += 1
        if self.filter:
            _ent = self.filter(_ent)
            if not _ent:
                flag = 0
        if flag:
            self.entity[entity_descr.entity_id] = _ent

    def parse(self, xmlstr):
        try:
            self.entities_descr = md.entities_descriptor_from_string(xmlstr)
        except Exception as e:
            _md_desc = f'metadata file: {self.filename}' if isinstance(self, MetaDataFile) else f'remote metadata: {self.url}' if isinstance(self, MetaDataExtern) else 'metadata'
            raise SAMLError(f'Failed to parse {_md_desc}') from e
        if not self.entities_descr:
            self.entity_descr = md.entity_descriptor_from_string(xmlstr)
            if self.entity_descr:
                self.do_entity_descriptor(self.entity_descr)
        else:
            try:
                valid_instance(self.entities_descr)
            except NotValid as exc:
                logger.error('Invalid XML message: %s', exc.args[0])
                return
            if self.check_validity:
                try:
                    if not valid(self.entities_descr.valid_until):
                        raise TooOld("Metadata not valid anymore, it's only valid until %s" % (self.entities_descr.valid_until,))
                except AttributeError:
                    pass
            for entity_descr in self.entities_descr.entity_descriptor:
                self.do_entity_descriptor(entity_descr)

    def service(self, entity_id, typ, service, binding=None):
        """Get me all services with a specified
        entity ID and type, that supports the specified version of binding.

        :param entity_id: The EntityId
        :param typ: Type of service (idp, attribute_authority, ...)
        :param service: which service that is sought for
        :param binding: A binding identifier
        :return: list of service descriptions.
            Or if no binding was specified a list of 2-tuples (binding, srv)
        """
        try:
            srvs = []
            for t in self[entity_id][typ]:
                try:
                    srvs.extend(t[service])
                except KeyError:
                    pass
        except KeyError:
            return None
        if not srvs:
            return srvs
        if binding:
            res = []
            for srv in srvs:
                if srv['binding'] == binding:
                    res.append(srv)
        else:
            res = {}
            for srv in srvs:
                try:
                    res[srv['binding']].append(srv)
                except KeyError:
                    res[srv['binding']] = [srv]
        logger.debug('service => %s', res)
        return res

    def attribute_requirement(self, entity_id, index=None):
        """
        Returns what attributes the SP requires and which are optional
        if any such demands are registered in the Metadata.

        In case the metadata have multiple SPSSODescriptor elements,
        the sum of the required and optional attributes is returned.

        :param entity_id: The entity id of the SP
        :param index: which of the attribute consumer services its all about
            if index=None then return all attributes expected by all
            attribute_consuming_services.
        :return: dict of required and optional list of attributes
        """
        res = {'required': [], 'optional': []}
        sp_descriptors = self[entity_id].get('spsso_descriptor') or []
        for sp_desc in sp_descriptors:
            _res = attribute_requirement(sp_desc, index)
            res['required'].extend(_res.get('required') or [])
            res['optional'].extend(_res.get('optional') or [])
        return res

    def construct_source_id(self):
        res = {}
        for eid, ent in self.items():
            for desc in ['spsso_descriptor', 'idpsso_descriptor']:
                try:
                    for srv in ent[desc]:
                        if 'artifact_resolution_service' in srv:
                            if isinstance(eid, str):
                                eid = eid.encode('utf-8')
                            s = sha1(eid)
                            res[s.digest()] = ent
                except KeyError:
                    pass
        return res

    def signed(self):
        if self.entities_descr and self.entities_descr.signature:
            return True
        if self.entity_descr and self.entity_descr.signature:
            return True
        else:
            return False

    def parse_and_check_signature(self, txt):
        self.parse(txt)
        if not self.cert:
            return True
        if not self.signed():
            return True
        if self.node_name is not None:
            try:
                self.security.verify_signature(txt, node_name=self.node_name, cert_file=self.cert)
            except SignatureError as e:
                error_context = {'message': 'Failed to verify signature', 'node_name': self.node_name}
                raise SignatureError(error_context) from e
            else:
                return True

        def try_verify_signature(node_name):
            try:
                self.security.verify_signature(txt, node_name=node_name, cert_file=self.cert)
            except SignatureError:
                return False
            else:
                return True
        descriptor_names = [f'{ns}:{tag}' for ns, tag in [(EntitiesDescriptor.c_namespace, EntitiesDescriptor.c_tag), (EntityDescriptor.c_namespace, EntityDescriptor.c_tag)]]
        verified_w_descriptor_name = any((try_verify_signature(node_name) for node_name in descriptor_names))
        if not verified_w_descriptor_name:
            error_context = {'message': 'Failed to verify signature', 'descriptor_names': descriptor_names}
            raise SignatureError(error_context)
        return verified_w_descriptor_name