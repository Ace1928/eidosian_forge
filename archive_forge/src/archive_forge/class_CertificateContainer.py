import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
class CertificateContainer(CertificateContainerFormatter, Container):
    _required_secrets = ['certificate', 'private_key']
    _optional_secrets = ['private_key_passphrase', 'intermediates']
    _type = 'certificate'

    def __init__(self, api, name=None, certificate=None, intermediates=None, private_key=None, private_key_passphrase=None, consumers=[], container_ref=None, created=None, updated=None, status=None, certificate_ref=None, intermediates_ref=None, private_key_ref=None, private_key_passphrase_ref=None):
        secret_refs = {}
        if certificate_ref:
            secret_refs['certificate'] = certificate_ref
        if intermediates_ref:
            secret_refs['intermediates'] = intermediates_ref
        if private_key_ref:
            secret_refs['private_key'] = private_key_ref
        if private_key_passphrase_ref:
            secret_refs['private_key_passphrase'] = private_key_passphrase_ref
        super(CertificateContainer, self).__init__(api=api, name=name, consumers=consumers, container_ref=container_ref, created=created, updated=updated, status=status, secret_refs=secret_refs)
        if certificate:
            self.certificate = certificate
        if intermediates:
            self.intermediates = intermediates
        if private_key:
            self.private_key = private_key
        if private_key_passphrase:
            self.private_key_passphrase = private_key_passphrase

    @property
    def certificate(self):
        """Secret containing the certificate"""
        return self._get_named_secret('certificate')

    @property
    def private_key(self):
        """Secret containing the private key"""
        return self._get_named_secret('private_key')

    @property
    def private_key_passphrase(self):
        """Secret containing the passphrase"""
        return self._get_named_secret('private_key_passphrase')

    @property
    def intermediates(self):
        """Secret containing intermediate certificates"""
        return self._get_named_secret('intermediates')

    @certificate.setter
    @_immutable_after_save
    def certificate(self, value):
        super(CertificateContainer, self).remove('certificate')
        super(CertificateContainer, self).add('certificate', value)

    @private_key.setter
    @_immutable_after_save
    def private_key(self, value):
        super(CertificateContainer, self).remove('private_key')
        super(CertificateContainer, self).add('private_key', value)

    @private_key_passphrase.setter
    @_immutable_after_save
    def private_key_passphrase(self, value):
        super(CertificateContainer, self).remove('private_key_passphrase')
        super(CertificateContainer, self).add('private_key_passphrase', value)

    @intermediates.setter
    @_immutable_after_save
    def intermediates(self, value):
        super(CertificateContainer, self).remove('intermediates')
        super(CertificateContainer, self).add('intermediates', value)

    def add(self, name, sec):
        raise NotImplementedError('`add()` is not implemented for Typed Containers')

    def __repr__(self):
        return 'CertificateContainer(name="{0}")'.format(self.name)