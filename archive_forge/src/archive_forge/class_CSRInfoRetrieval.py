from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
@six.add_metaclass(abc.ABCMeta)
class CSRInfoRetrieval(object):

    def __init__(self, module, backend, content, validate_signature):
        self.module = module
        self.backend = backend
        self.content = content
        self.validate_signature = validate_signature

    @abc.abstractmethod
    def _get_subject_ordered(self):
        pass

    @abc.abstractmethod
    def _get_key_usage(self):
        pass

    @abc.abstractmethod
    def _get_extended_key_usage(self):
        pass

    @abc.abstractmethod
    def _get_basic_constraints(self):
        pass

    @abc.abstractmethod
    def _get_ocsp_must_staple(self):
        pass

    @abc.abstractmethod
    def _get_subject_alt_name(self):
        pass

    @abc.abstractmethod
    def _get_name_constraints(self):
        pass

    @abc.abstractmethod
    def _get_public_key_pem(self):
        pass

    @abc.abstractmethod
    def _get_public_key_object(self):
        pass

    @abc.abstractmethod
    def _get_subject_key_identifier(self):
        pass

    @abc.abstractmethod
    def _get_authority_key_identifier(self):
        pass

    @abc.abstractmethod
    def _get_all_extensions(self):
        pass

    @abc.abstractmethod
    def _is_signature_valid(self):
        pass

    def get_info(self, prefer_one_fingerprint=False):
        result = dict()
        self.csr = load_certificate_request(None, content=self.content, backend=self.backend)
        subject = self._get_subject_ordered()
        result['subject'] = dict()
        for k, v in subject:
            result['subject'][k] = v
        result['subject_ordered'] = subject
        result['key_usage'], result['key_usage_critical'] = self._get_key_usage()
        result['extended_key_usage'], result['extended_key_usage_critical'] = self._get_extended_key_usage()
        result['basic_constraints'], result['basic_constraints_critical'] = self._get_basic_constraints()
        result['ocsp_must_staple'], result['ocsp_must_staple_critical'] = self._get_ocsp_must_staple()
        result['subject_alt_name'], result['subject_alt_name_critical'] = self._get_subject_alt_name()
        result['name_constraints_permitted'], result['name_constraints_excluded'], result['name_constraints_critical'] = self._get_name_constraints()
        result['public_key'] = to_native(self._get_public_key_pem())
        public_key_info = get_publickey_info(self.module, self.backend, key=self._get_public_key_object(), prefer_one_fingerprint=prefer_one_fingerprint)
        result.update({'public_key_type': public_key_info['type'], 'public_key_data': public_key_info['public_data'], 'public_key_fingerprints': public_key_info['fingerprints']})
        ski = self._get_subject_key_identifier()
        if ski is not None:
            ski = to_native(binascii.hexlify(ski))
            ski = ':'.join([ski[i:i + 2] for i in range(0, len(ski), 2)])
        result['subject_key_identifier'] = ski
        aki, aci, acsn = self._get_authority_key_identifier()
        if aki is not None:
            aki = to_native(binascii.hexlify(aki))
            aki = ':'.join([aki[i:i + 2] for i in range(0, len(aki), 2)])
        result['authority_key_identifier'] = aki
        result['authority_cert_issuer'] = aci
        result['authority_cert_serial_number'] = acsn
        result['extensions_by_oid'] = self._get_all_extensions()
        result['signature_valid'] = self._is_signature_valid()
        if self.validate_signature and (not result['signature_valid']):
            self.module.fail_json(msg='CSR signature is invalid!', **result)
        return result