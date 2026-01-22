from __future__ import absolute_import, division, print_function
import abc
import binascii
import datetime
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
class CertificateInfoRetrievalCryptography(CertificateInfoRetrieval):
    """Validate the supplied cert, using the cryptography backend"""

    def __init__(self, module, content):
        super(CertificateInfoRetrievalCryptography, self).__init__(module, 'cryptography', content)
        self.name_encoding = module.params.get('name_encoding', 'ignore')

    def _get_der_bytes(self):
        return self.cert.public_bytes(serialization.Encoding.DER)

    def _get_signature_algorithm(self):
        return cryptography_oid_to_name(self.cert.signature_algorithm_oid)

    def _get_subject_ordered(self):
        result = []
        for attribute in self.cert.subject:
            result.append([cryptography_oid_to_name(attribute.oid), attribute.value])
        return result

    def _get_issuer_ordered(self):
        result = []
        for attribute in self.cert.issuer:
            result.append([cryptography_oid_to_name(attribute.oid), attribute.value])
        return result

    def _get_version(self):
        if self.cert.version == x509.Version.v1:
            return 1
        if self.cert.version == x509.Version.v3:
            return 3
        return 'unknown'

    def _get_key_usage(self):
        try:
            current_key_ext = self.cert.extensions.get_extension_for_class(x509.KeyUsage)
            current_key_usage = current_key_ext.value
            key_usage = dict(digital_signature=current_key_usage.digital_signature, content_commitment=current_key_usage.content_commitment, key_encipherment=current_key_usage.key_encipherment, data_encipherment=current_key_usage.data_encipherment, key_agreement=current_key_usage.key_agreement, key_cert_sign=current_key_usage.key_cert_sign, crl_sign=current_key_usage.crl_sign, encipher_only=False, decipher_only=False)
            if key_usage['key_agreement']:
                key_usage.update(dict(encipher_only=current_key_usage.encipher_only, decipher_only=current_key_usage.decipher_only))
            key_usage_names = dict(digital_signature='Digital Signature', content_commitment='Non Repudiation', key_encipherment='Key Encipherment', data_encipherment='Data Encipherment', key_agreement='Key Agreement', key_cert_sign='Certificate Sign', crl_sign='CRL Sign', encipher_only='Encipher Only', decipher_only='Decipher Only')
            return (sorted([key_usage_names[name] for name, value in key_usage.items() if value]), current_key_ext.critical)
        except cryptography.x509.ExtensionNotFound:
            return (None, False)

    def _get_extended_key_usage(self):
        try:
            ext_keyusage_ext = self.cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
            return (sorted([cryptography_oid_to_name(eku) for eku in ext_keyusage_ext.value]), ext_keyusage_ext.critical)
        except cryptography.x509.ExtensionNotFound:
            return (None, False)

    def _get_basic_constraints(self):
        try:
            ext_keyusage_ext = self.cert.extensions.get_extension_for_class(x509.BasicConstraints)
            result = []
            result.append('CA:{0}'.format('TRUE' if ext_keyusage_ext.value.ca else 'FALSE'))
            if ext_keyusage_ext.value.path_length is not None:
                result.append('pathlen:{0}'.format(ext_keyusage_ext.value.path_length))
            return (sorted(result), ext_keyusage_ext.critical)
        except cryptography.x509.ExtensionNotFound:
            return (None, False)

    def _get_ocsp_must_staple(self):
        try:
            try:
                tlsfeature_ext = self.cert.extensions.get_extension_for_class(x509.TLSFeature)
                value = cryptography.x509.TLSFeatureType.status_request in tlsfeature_ext.value
            except AttributeError:
                oid = x509.oid.ObjectIdentifier('1.3.6.1.5.5.7.1.24')
                tlsfeature_ext = self.cert.extensions.get_extension_for_oid(oid)
                value = tlsfeature_ext.value.value == b'0\x03\x02\x01\x05'
            return (value, tlsfeature_ext.critical)
        except cryptography.x509.ExtensionNotFound:
            return (None, False)

    def _get_subject_alt_name(self):
        try:
            san_ext = self.cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            result = [cryptography_decode_name(san, idn_rewrite=self.name_encoding) for san in san_ext.value]
            return (result, san_ext.critical)
        except cryptography.x509.ExtensionNotFound:
            return (None, False)

    def get_not_before(self):
        return self.cert.not_valid_before

    def get_not_after(self):
        return self.cert.not_valid_after

    def _get_public_key_pem(self):
        return self.cert.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    def _get_public_key_object(self):
        return self.cert.public_key()

    def _get_subject_key_identifier(self):
        try:
            ext = self.cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
            return ext.value.digest
        except cryptography.x509.ExtensionNotFound:
            return None

    def _get_authority_key_identifier(self):
        try:
            ext = self.cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
            issuer = None
            if ext.value.authority_cert_issuer is not None:
                issuer = [cryptography_decode_name(san, idn_rewrite=self.name_encoding) for san in ext.value.authority_cert_issuer]
            return (ext.value.key_identifier, issuer, ext.value.authority_cert_serial_number)
        except cryptography.x509.ExtensionNotFound:
            return (None, None, None)

    def _get_serial_number(self):
        return cryptography_serial_number_of_cert(self.cert)

    def _get_all_extensions(self):
        return cryptography_get_extensions_from_cert(self.cert)

    def _get_ocsp_uri(self):
        try:
            ext = self.cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
            for desc in ext.value:
                if desc.access_method == x509.oid.AuthorityInformationAccessOID.OCSP:
                    if isinstance(desc.access_location, x509.UniformResourceIdentifier):
                        return desc.access_location.value
        except x509.ExtensionNotFound as dummy:
            pass
        return None

    def _get_issuer_uri(self):
        try:
            ext = self.cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess)
            for desc in ext.value:
                if desc.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS:
                    if isinstance(desc.access_location, x509.UniformResourceIdentifier):
                        return desc.access_location.value
        except x509.ExtensionNotFound as dummy:
            pass
        return None