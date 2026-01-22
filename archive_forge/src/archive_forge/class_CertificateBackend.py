from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_info import (
@six.add_metaclass(abc.ABCMeta)
class CertificateBackend(object):

    def __init__(self, module, backend):
        self.module = module
        self.backend = backend
        self.force = module.params['force']
        self.ignore_timestamps = module.params['ignore_timestamps']
        self.privatekey_path = module.params['privatekey_path']
        self.privatekey_content = module.params['privatekey_content']
        if self.privatekey_content is not None:
            self.privatekey_content = self.privatekey_content.encode('utf-8')
        self.privatekey_passphrase = module.params['privatekey_passphrase']
        self.csr_path = module.params['csr_path']
        self.csr_content = module.params['csr_content']
        if self.csr_content is not None:
            self.csr_content = self.csr_content.encode('utf-8')
        self.create_subject_key_identifier = 'never_create'
        self.create_authority_key_identifier = False
        self.privatekey = None
        self.csr = None
        self.cert = None
        self.existing_certificate = None
        self.existing_certificate_bytes = None
        self.check_csr_subject = True
        self.check_csr_extensions = True
        self.diff_before = self._get_info(None)
        self.diff_after = self._get_info(None)

    def _get_info(self, data):
        if data is None:
            return dict()
        try:
            result = get_certificate_info(self.module, self.backend, data, prefer_one_fingerprint=True)
            result['can_parse_certificate'] = True
            return result
        except Exception as exc:
            return dict(can_parse_certificate=False)

    @abc.abstractmethod
    def generate_certificate(self):
        """(Re-)Generate certificate."""
        pass

    @abc.abstractmethod
    def get_certificate_data(self):
        """Return bytes for self.cert."""
        pass

    def set_existing(self, certificate_bytes):
        """Set existing certificate bytes. None indicates that the key does not exist."""
        self.existing_certificate_bytes = certificate_bytes
        self.diff_after = self.diff_before = self._get_info(self.existing_certificate_bytes)

    def has_existing(self):
        """Query whether an existing certificate is/has been there."""
        return self.existing_certificate_bytes is not None

    def _ensure_private_key_loaded(self):
        """Load the provided private key into self.privatekey."""
        if self.privatekey is not None:
            return
        if self.privatekey_path is None and self.privatekey_content is None:
            return
        try:
            self.privatekey = load_privatekey(path=self.privatekey_path, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
        except OpenSSLBadPassphraseError as exc:
            raise CertificateError(exc)

    def _ensure_csr_loaded(self):
        """Load the CSR into self.csr."""
        if self.csr is not None:
            return
        if self.csr_path is None and self.csr_content is None:
            return
        self.csr = load_certificate_request(path=self.csr_path, content=self.csr_content, backend=self.backend)

    def _ensure_existing_certificate_loaded(self):
        """Load the existing certificate into self.existing_certificate."""
        if self.existing_certificate is not None:
            return
        if self.existing_certificate_bytes is None:
            return
        self.existing_certificate = load_certificate(path=None, content=self.existing_certificate_bytes, backend=self.backend)

    def _check_privatekey(self):
        """Check whether provided parameters match, assuming self.existing_certificate and self.privatekey have been populated."""
        if self.backend == 'cryptography':
            return cryptography_compare_public_keys(self.existing_certificate.public_key(), self.privatekey.public_key())

    def _check_csr(self):
        """Check whether provided parameters match, assuming self.existing_certificate and self.csr have been populated."""
        if self.backend == 'cryptography':
            if not self.csr.is_signature_valid:
                return False
            if not cryptography_compare_public_keys(self.csr.public_key(), self.existing_certificate.public_key()):
                return False
            if self.check_csr_subject and self.csr.subject != self.existing_certificate.subject:
                return False
            if not self.check_csr_extensions:
                return True
            cert_exts = list(self.existing_certificate.extensions)
            csr_exts = list(self.csr.extensions)
            if self.create_subject_key_identifier != 'never_create':
                cert_exts = list(filter(lambda x: not isinstance(x.value, x509.SubjectKeyIdentifier), cert_exts))
                csr_exts = list(filter(lambda x: not isinstance(x.value, x509.SubjectKeyIdentifier), csr_exts))
            if self.create_authority_key_identifier:
                cert_exts = list(filter(lambda x: not isinstance(x.value, x509.AuthorityKeyIdentifier), cert_exts))
                csr_exts = list(filter(lambda x: not isinstance(x.value, x509.AuthorityKeyIdentifier), csr_exts))
            if len(cert_exts) != len(csr_exts):
                return False
            for cert_ext in cert_exts:
                try:
                    csr_ext = self.csr.extensions.get_extension_for_oid(cert_ext.oid)
                    if cert_ext != csr_ext:
                        return False
                except cryptography.x509.ExtensionNotFound as dummy:
                    return False
            return True

    def _check_subject_key_identifier(self):
        """Check whether Subject Key Identifier matches, assuming self.existing_certificate has been populated."""
        try:
            ext = self.existing_certificate.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        except cryptography.x509.ExtensionNotFound as dummy:
            return False
        csr_ext = None
        if self.create_subject_key_identifier == 'create_if_not_provided':
            try:
                csr_ext = self.csr.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
            except cryptography.x509.ExtensionNotFound as dummy:
                pass
        if csr_ext is None:
            if ext.value.digest != x509.SubjectKeyIdentifier.from_public_key(self.existing_certificate.public_key()).digest:
                return False
        elif ext.value.digest != csr_ext.value.digest:
            return False
        return True

    def needs_regeneration(self, not_before=None, not_after=None):
        """Check whether a regeneration is necessary."""
        if self.force or self.existing_certificate_bytes is None:
            return True
        try:
            self._ensure_existing_certificate_loaded()
        except Exception as dummy:
            return True
        self._ensure_private_key_loaded()
        if self.privatekey is not None and (not self._check_privatekey()):
            return True
        self._ensure_csr_loaded()
        if self.csr is not None and (not self._check_csr()):
            return True
        if self.create_subject_key_identifier != 'never_create' and (not self._check_subject_key_identifier()):
            return True
        if not_before is not None and (not self.ignore_timestamps):
            if self.existing_certificate.not_valid_before != not_before:
                return True
        if not_after is not None and (not self.ignore_timestamps):
            if self.existing_certificate.not_valid_after != not_after:
                return True
        return False

    def dump(self, include_certificate):
        """Serialize the object into a dictionary."""
        result = {'privatekey': self.privatekey_path, 'csr': self.csr_path}
        certificate_bytes = self.existing_certificate_bytes
        if self.cert is not None:
            certificate_bytes = self.get_certificate_data()
        self.diff_after = self._get_info(certificate_bytes)
        if include_certificate:
            result['certificate'] = certificate_bytes.decode('utf-8') if certificate_bytes else None
        result['diff'] = dict(before=self.diff_before, after=self.diff_after)
        return result