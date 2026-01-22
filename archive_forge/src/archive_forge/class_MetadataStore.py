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
class MetadataStore(MetaData):

    def __init__(self, attrc, config, ca_certs=None, check_validity=True, disable_ssl_certificate_validation=False, filter=None, http_client_timeout=None):
        """
        :params attrc:
        :params config: Config()
        :params ca_certs:
        :params disable_ssl_certificate_validation:
        """
        MetaData.__init__(self, attrc, check_validity=check_validity)
        if disable_ssl_certificate_validation:
            self.http = HTTPBase(verify=False, ca_bundle=ca_certs, http_client_timeout=http_client_timeout)
        else:
            self.http = HTTPBase(verify=True, ca_bundle=ca_certs, http_client_timeout=http_client_timeout)
        self.security = security_context(config)
        self.ii = 0
        self.metadata = {}
        self.check_validity = check_validity
        self.filter = filter
        self.to_old = {}
        self.http_client_timeout = http_client_timeout

    def load(self, *args, **kwargs):
        if self.filter:
            _args = {'filter': self.filter}
        else:
            _args = {}
        typ = args[0]
        if typ == 'local':
            key = args[1]
            if os.path.isdir(key):
                files = [f for f in os.listdir(key) if isfile(join(key, f))]
                for fil in files:
                    _fil = join(key, fil)
                    _md = MetaDataFile(self.attrc, _fil, **_args)
                    _md.load()
                    self.metadata[_fil] = _md
                return
            else:
                _md = MetaDataFile(self.attrc, key, **_args)
        elif typ == 'inline':
            self.ii += 1
            key = self.ii
            kwargs.update(_args)
            _md = InMemoryMetaData(self.attrc, args[1])
        elif typ == 'remote':
            if 'url' not in kwargs:
                raise ValueError("Remote metadata must be structured as a dict containing the key 'url'")
            key = kwargs['url']
            for _key in ['node_name', 'check_validity']:
                try:
                    _args[_key] = kwargs[_key]
                except KeyError:
                    pass
            if 'cert' not in kwargs:
                kwargs['cert'] = ''
            _md = MetaDataExtern(self.attrc, kwargs['url'], self.security, kwargs['cert'], self.http, **_args)
        elif typ == 'mdfile':
            key = args[1]
            _md = MetaDataMD(self.attrc, args[1], **_args)
        elif typ == 'loader':
            key = args[1]
            _md = MetaDataLoader(self.attrc, args[1], **_args)
        elif typ == 'mdq':
            if 'url' in kwargs:
                key = kwargs['url']
                url = kwargs['url']
                cert = kwargs.get('cert')
                freshness_period = kwargs.get('freshness_period', None)
                security = self.security
                entity_transform = kwargs.get('entity_transform', None)
                _md = MetaDataMDX(url, security, cert, entity_transform, freshness_period=freshness_period, http_client_timeout=self.http_client_timeout)
            else:
                key = args[1]
                url = args[1]
                _md = MetaDataMDX(url, http_client_timeout=self.http_client_timeout)
        else:
            raise SAMLError(f"Unknown metadata type '{typ}'")
        _md.load()
        self.metadata[key] = _md

    def reload(self, spec):
        old_metadata = self.metadata
        self.metadata = {}
        try:
            self.imp(spec)
        except Exception as e:
            self.metadata = old_metadata
            raise e

    def imp(self, spec):
        if type(spec) is dict:
            for key, vals in spec.items():
                for val in vals:
                    if isinstance(val, dict):
                        if not self.check_validity:
                            val['check_validity'] = False
                        self.load(key, **val)
                    else:
                        self.load(key, val)
        else:
            for item in spec:
                try:
                    key = item['class']
                except (KeyError, AttributeError):
                    raise SAMLError(f'Misconfiguration in metadata {item}')
                mod, clas = key.rsplit('.', 1)
                try:
                    mod = importlib.import_module(mod)
                    MDloader = getattr(mod, clas)
                except (ImportError, AttributeError):
                    raise SAMLError(f'Unknown metadata loader {key}')
                if MDloader == MetaDataExtern:
                    kwargs = {'http': self.http, 'security': self.security}
                else:
                    kwargs = {}
                if self.filter:
                    kwargs['filter'] = self.filter
                for key in item['metadata']:
                    if MDloader == MetaDataFile and os.path.isdir(key[0]):
                        files = [f for f in os.listdir(key[0]) if isfile(join(key[0], f))]
                        for fil in files:
                            _fil = join(key[0], fil)
                            _md = MetaDataFile(self.attrc, _fil)
                            _md.load()
                            self.metadata[_fil] = _md
                            if _md.to_old:
                                self.to_old[_fil] = _md.to_old
                        return
                    if len(key) == 2:
                        kwargs['cert'] = key[1]
                    _md = MDloader(self.attrc, key[0], **kwargs)
                    _md.load()
                    self.metadata[key[0]] = _md
                    if _md.to_old:
                        self.to_old[key[0]] = _md.to_old

    def service(self, entity_id, typ, service, binding=None):
        known_entity = False
        logger.debug('service(%s, %s, %s, %s)', entity_id, typ, service, binding)
        for key, _md in self.metadata.items():
            srvs = _md.service(entity_id, typ, service, binding)
            if srvs:
                return srvs
            elif srvs is None:
                pass
            else:
                known_entity = True
        if known_entity:
            logger.error('Unsupported binding: %s (%s)', binding, entity_id)
            raise UnsupportedBinding(binding)
        else:
            logger.error('Unknown system entity: %s', entity_id)
            raise UnknownSystemEntity(entity_id)

    def extension(self, entity_id, typ, service):
        for key, _md in self.metadata.items():
            try:
                srvs = _md[entity_id][typ]
            except KeyError:
                continue
            res = []
            for srv in srvs:
                if 'extensions' in srv:
                    for elem in srv['extensions']['extension_elements']:
                        if elem['__class__'] == service:
                            res.append(elem)
            return res
        return None

    def ext_service(self, entity_id, typ, service, binding=None):
        known_entity = False
        for key, _md in self.metadata.items():
            srvs = _md.ext_service(entity_id, typ, service, binding)
            if srvs:
                return srvs
            elif srvs is None:
                pass
            else:
                known_entity = True
        if known_entity:
            raise UnsupportedBinding(binding)
        else:
            raise UnknownSystemEntity(entity_id)

    def single_sign_on_service(self, entity_id, binding=None, typ='idpsso'):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, 'idpsso_descriptor', 'single_sign_on_service', binding)

    def name_id_mapping_service(self, entity_id, binding=None, typ='idpsso'):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, 'idpsso_descriptor', 'name_id_mapping_service', binding)

    def authn_query_service(self, entity_id, binding=None, typ='authn_authority'):
        if binding is None:
            binding = BINDING_SOAP
        return self.service(entity_id, 'authn_authority_descriptor', 'authn_query_service', binding)

    def attribute_service(self, entity_id, binding=None, typ='attribute_authority'):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, 'attribute_authority_descriptor', 'attribute_service', binding)

    def authz_service(self, entity_id, binding=None, typ='pdp'):
        if binding is None:
            binding = BINDING_SOAP
        return self.service(entity_id, 'pdp_descriptor', 'authz_service', binding)

    def assertion_id_request_service(self, entity_id, binding=None, typ=None):
        if typ is None:
            raise AttributeError('Missing type specification')
        if binding is None:
            binding = BINDING_SOAP
        return self.service(entity_id, f'{typ}_descriptor', 'assertion_id_request_service', binding)

    def single_logout_service(self, entity_id, binding=None, typ=None):
        if typ is None:
            raise AttributeError('Missing type specification')
        return self.service(entity_id, f'{typ}_descriptor', 'single_logout_service', binding)

    def manage_name_id_service(self, entity_id, binding=None, typ=None):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, f'{typ}_descriptor', 'manage_name_id_service', binding)

    def artifact_resolution_service(self, entity_id, binding=None, typ=None):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, f'{typ}_descriptor', 'artifact_resolution_service', binding)

    def assertion_consumer_service(self, entity_id, binding=None, _='spsso'):
        if binding is None:
            binding = BINDING_HTTP_POST
        return self.service(entity_id, 'spsso_descriptor', 'assertion_consumer_service', binding)

    def attribute_consuming_service(self, entity_id, binding=None, _='spsso'):
        if binding is None:
            binding = BINDING_HTTP_REDIRECT
        return self.service(entity_id, 'spsso_descriptor', 'attribute_consuming_service', binding)

    def discovery_response(self, entity_id, binding=None, _='spsso'):
        if binding is None:
            binding = BINDING_DISCO
        return self.ext_service(entity_id, 'spsso_descriptor', f'{DiscoveryResponse.c_namespace}&{DiscoveryResponse.c_tag}', binding)

    def attribute_requirement(self, entity_id, index=None):
        for md_source in self.metadata.values():
            if entity_id in md_source:
                return md_source.attribute_requirement(entity_id, index)

    def subject_id_requirement(self, entity_id):
        try:
            entity_attributes = self.entity_attributes(entity_id)
        except KeyError:
            return []
        subject_id_reqs = entity_attributes.get('urn:oasis:names:tc:SAML:profiles:subject-id:req') or []
        subject_id_req = next(iter(subject_id_reqs), None)
        if subject_id_req == 'any':
            return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:pairwise-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'pairwise-id', 'is_required': 'true'}, {'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:subject-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'subject-id', 'is_required': 'true'}]
        elif subject_id_req == 'pairwise-id':
            return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:pairwise-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'pairwise-id', 'is_required': 'true'}]
        elif subject_id_req == 'subject-id':
            return [{'__class__': 'urn:oasis:names:tc:SAML:2.0:metadata&RequestedAttribute', 'name': 'urn:oasis:names:tc:SAML:attribute:subject-id', 'name_format': 'urn:oasis:names:tc:SAML:2.0:attrname-format:uri', 'friendly_name': 'subject-id', 'is_required': 'true'}]
        return []

    def keys(self):
        res = []
        for _md in self.metadata.values():
            res.extend(_md.keys())
        return res

    def __getitem__(self, item):
        for _md in self.metadata.values():
            try:
                return _md[item]
            except KeyError:
                pass
        raise KeyError(item)

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def entities(self):
        num = 0
        for _md in self.metadata.values():
            num += len(_md.items())
        return num

    def __len__(self):
        return len(self.metadata)

    def with_descriptor(self, descriptor):
        res = {}
        for _md in self.metadata.values():
            res.update(_md.with_descriptor(descriptor))
        return res

    def name(self, entity_id, langpref='en'):
        for _md in self.metadata.values():
            if entity_id in _md:
                return name(_md[entity_id], langpref)
        return None

    def vo_members(self, entity_id):
        ad = self.__getitem__(entity_id)['affiliation_descriptor']
        return [m['text'] for m in ad['affiliate_member']]

    def entity_categories(self, entity_id):
        """
        Get a list of entity categories for an entity id.

        :param entity_id: Entity id
        :return: Entity categories

        :type entity_id: string
        :rtype: [string]
        """
        attributes = self.entity_attributes(entity_id)
        return attributes.get(ENTITY_CATEGORY, [])

    def supported_entity_categories(self, entity_id):
        """
        Get a list of entity category support for an entity id.

        :param entity_id: Entity id
        :return: Entity category support

        :type entity_id: string
        :rtype: [string]
        """
        attributes = self.entity_attributes(entity_id)
        return attributes.get(ENTITY_CATEGORY_SUPPORT, [])

    def assurance_certifications(self, entity_id):
        assurance_certifications = (certification for name, values in self.entity_attributes(entity_id).items() if name == ASSURANCE_CERTIFICATION for certification in values)
        return assurance_certifications

    def entity_attributes(self, entity_id):
        """
        Get all entity attributes for an entry in the metadata.

        Example return data:

        {'http://macedir.org/entity-category': ['something', 'something2'],
         'http://example.org/saml-foo': ['bar']}

        :param entity_id: Entity id
        :return: dict with keys and value-lists from metadata

        :type entity_id: string
        :rtype: dict
        """
        res = {}
        try:
            ext = self.__getitem__(entity_id)['extensions']
        except KeyError:
            return res
        for elem in ext['extension_elements']:
            if elem['__class__'] != classnames['mdattr_entityattributes']:
                continue
            for attr in elem['attribute']:
                res[attr['name']] = [*res.get(attr['name'], []), *(v['text'] for v in attr.get('attribute_value', []))]
        return res

    def supported_algorithms(self, entity_id):
        """
        Get all supported algorithms for an entry in the metadata.

        Example return data:

        {'digest_methods': ['http://www.w3.org/2001/04/xmldsig-more#sha224', 'http://www.w3.org/2001/04/xmlenc#sha256'],
         'signing_methods': ['http://www.w3.org/2001/04/xmldsig-more#rsa-sha256']}

        :param entity_id: Entity id
        :return: dict with keys and value-lists from metadata

        :type entity_id: string
        :rtype: dict
        """
        res = {'digest_methods': [], 'signing_methods': []}
        try:
            ext = self.__getitem__(entity_id)['extensions']
        except KeyError:
            return res
        for elem in ext['extension_elements']:
            if elem['__class__'] == classnames['algsupport_digest_method']:
                res['digest_methods'].append(elem['algorithm'])
            elif elem['__class__'] == classnames['algsupport_signing_method']:
                res['signing_methods'].append(elem['algorithm'])
        return res

    def registration_info(self, entity_id):
        """
        Get all registration info for an entry in the metadata.

        Example return data:

        res = {
            'registration_authority': 'http://www.example.com',
            'registration_instant': '2013-06-15T18:15:03Z',
            'registration_policy': {
                'en': 'http://www.example.com/policy.html',
                'sv': 'http://www.example.com/sv/policy.html',
            }
        }

        :param entity_id: Entity id
        :return: dict with keys and value-lists from metadata

        :type entity_id: string
        :rtype: dict
        """
        try:
            ext = self.__getitem__(entity_id)
        except KeyError:
            ext = {}
        ext_elems = ext.get('extensions', {}).get('extension_elements', [])
        reg_info = next((elem for elem in ext_elems if elem['__class__'] == classnames['mdrpi_registration_info']), {})
        res = {'registration_authority': reg_info.get('registration_authority'), 'registration_instant': reg_info.get('registration_instant'), 'registration_policy': {policy['lang']: policy['text'] for policy in reg_info.get('registration_policy', []) if policy['__class__'] == classnames['mdrpi_registration_policy']}}
        return res

    def registration_info_typ(self, entity_id, typ):
        try:
            md = self.__getitem__(entity_id)
        except KeyError:
            md = {}
        services_of_type = md.get(typ) or []
        typ_reg_info = ({'registration_authority': elem.get('registration_authority'), 'registration_instant': elem.get('registration_instant'), 'registration_policy': {policy['lang']: policy['text'] for policy in elem.get('registration_policy', []) if policy.get('__class__') == classnames['mdrpi_registration_policy']}} for srv in services_of_type for elem in srv.get('extensions', {}).get('extension_elements', []) if elem.get('__class__') == classnames['mdrpi_registration_info'])
        return typ_reg_info

    def _lookup_elements_by_cls(self, root, cls):
        elements = (element for uiinfo in root for element_key, elements in uiinfo.items() if element_key != '__class__' for element in elements if element.get('__class__') == cls)
        return elements

    def _lookup_elements_by_key(self, root, key):
        elements = (element for uiinfo in root for elements in [uiinfo.get(key, [])] for element in elements)
        return elements

    def sbibmd_scopes(self, entity_id, typ=None):
        warn_msg = '`saml2.mdstore.MetadataStore::sbibmd_scopes` method is deprecated; instead, use `saml2.mdstore.MetadataStore::shibmd_scopes`.'
        logger.warning(warn_msg)
        _warn(warn_msg, DeprecationWarning)
        return self.shibmd_scopes(entity_id, typ=typ)

    def shibmd_scopes(self, entity_id, typ=None):
        try:
            md = self[entity_id]
        except KeyError:
            md = {}
        descriptor_scopes = ({'regexp': is_regexp, 'text': regex_compile(text) if is_regexp else text} for elem in md.get('extensions', {}).get('extension_elements', []) if elem.get('__class__') == classnames['shibmd_scope'] for is_regexp, text in [(elem.get('regexp', '').lower() == 'true', elem.get('text', ''))])
        services_of_type = md.get(typ) or []
        services_of_type_scopes = ({'regexp': is_regexp, 'text': regex_compile(text) if is_regexp else text} for srv in services_of_type for elem in srv.get('extensions', {}).get('extension_elements', []) if elem.get('__class__') == classnames['shibmd_scope'] for is_regexp, text in [(elem.get('regexp', '').lower() == 'true', elem.get('text', ''))])
        scopes = chain(descriptor_scopes, services_of_type_scopes)
        return scopes

    def mdui_uiinfo(self, entity_id):
        try:
            data = self[entity_id]
        except KeyError:
            data = {}
        descriptor_names = (item for item in data.keys() if item.endswith('_descriptor'))
        descriptors = (descriptor for descriptor_name in descriptor_names for descriptor in self[entity_id].get(descriptor_name, []))
        extensions = (extension for descriptor in descriptors for extension in descriptor.get('extensions', {}).get('extension_elements', []))
        uiinfos = (extension for extension in extensions if extension.get('__class__') == classnames['mdui_uiinfo'])
        return uiinfos

    def _mdui_uiinfo_i18n_elements_lookup(self, entity_id, langpref, element_hint, lookup):
        uiinfos = self.mdui_uiinfo(entity_id)
        elements = lookup(uiinfos, element_hint)
        lang_elements = (element for element in elements if langpref is None or element.get('lang') == langpref)
        values = (value for element in lang_elements for value in [element.get('text')])
        return values

    def mdui_uiinfo_i18n_element_cls(self, entity_id, langpref, element_cls):
        values = self._mdui_uiinfo_i18n_elements_lookup(entity_id, langpref, element_cls, self._lookup_elements_by_cls)
        return values

    def mdui_uiinfo_i18n_element_key(self, entity_id, langpref, element_key):
        values = self._mdui_uiinfo_i18n_elements_lookup(entity_id, langpref, element_key, self._lookup_elements_by_key)
        return values

    def mdui_uiinfo_display_name(self, entity_id, langpref=None):
        cls = classnames['mdui_uiinfo_display_name']
        values = self.mdui_uiinfo_i18n_element_cls(entity_id, langpref, cls)
        return values

    def mdui_uiinfo_description(self, entity_id, langpref=None):
        cls = classnames['mdui_uiinfo_description']
        values = self.mdui_uiinfo_i18n_element_cls(entity_id, langpref, cls)
        return values

    def mdui_uiinfo_information_url(self, entity_id, langpref=None):
        cls = classnames['mdui_uiinfo_information_url']
        values = self.mdui_uiinfo_i18n_element_cls(entity_id, langpref, cls)
        return values

    def mdui_uiinfo_privacy_statement_url(self, entity_id, langpref=None):
        cls = classnames['mdui_uiinfo_privacy_statement_url']
        values = self.mdui_uiinfo_i18n_element_cls(entity_id, langpref, cls)
        return values

    def mdui_uiinfo_logo(self, entity_id, width=None, height=None):
        uiinfos = self.mdui_uiinfo(entity_id)
        cls = classnames['mdui_uiinfo_logo']
        elements = self._lookup_elements_by_cls(uiinfos, cls)
        values = (element for element in elements if width is None or element.get('width') == width if height is None or element.get('height') == height)
        return values

    def contact_person_data(self, entity_id, contact_type=None):
        try:
            data = self[entity_id]
        except KeyError:
            data = {}
        contacts = ({'contact_type': _contact_type, 'given_name': contact.get('given_name', {}).get('text', ''), 'email_address': [address for email in contact.get('email_address', {}) for address in [email.get('text')] if address]} for contact in data.get('contact_person', []) for _contact_type in [contact.get('contact_type', '')] if contact_type is None or contact_type == _contact_type)
        return contacts

    def bindings(self, entity_id, typ, service):
        for _md in self.metadata.values():
            if entity_id in _md.items():
                return _md.bindings(entity_id, typ, service)
        return None

    def __str__(self):
        _str = ['{']
        for key, val in self.metadata.items():
            _str.append(f'{key}: {val}')
        _str.append('}')
        return '\n'.join(_str)

    def construct_source_id(self):
        res = {}
        for _md in self.metadata.values():
            res.update(_md.construct_source_id())
        return res

    def items(self):
        res = {}
        for _md in self.metadata.values():
            res.update(_md.items())
        return res.items()

    def _providers(self, descriptor):
        res = []
        for _md in self.metadata.values():
            for ent_id, ent_desc in _md.items():
                if descriptor in ent_desc:
                    if ent_id in res:
                        pass
                    else:
                        res.append(ent_id)
        return res

    def service_providers(self):
        return self._providers('spsso_descriptor')

    def identity_providers(self):
        return self._providers('idpsso_descriptor')

    def attribute_authorities(self):
        return self._providers('attribute_authority')

    def dumps(self, format='local'):
        """
        Dumps the content in standard metadata format or the pysaml2 metadata
        format

        :param format: Which format to dump in
        :return: a string
        """
        if format == 'local':
            res = EntitiesDescriptor()
            for _md in self.metadata.values():
                try:
                    res.entity_descriptor.extend(_md.entities_descr.entity_descriptor)
                except AttributeError:
                    res.entity_descriptor.append(_md.entity_descr)
            return f'{res}'
        elif format == 'md':
            return json.dumps(dict(self.items()), indent=2)