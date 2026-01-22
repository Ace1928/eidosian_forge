from __future__ import absolute_import, division, print_function
import os
class HashiVaultAuthMethodBase(HashiVaultOptionGroupBase):
    """Base class for individual auth method implementations"""

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodBase, self).__init__(option_adapter)
        self._warner = warning_callback
        self._deprecator = deprecate_callback

    def validate(self):
        """Validates the given auth method as much as possible without calling Vault."""
        raise NotImplementedError('validate must be implemented')

    def authenticate(self, client, use_token=True):
        """Authenticates against Vault, returns a token."""
        raise NotImplementedError('authenticate must be implemented')

    def validate_by_required_fields(self, *field_names):
        missing = [field for field in field_names if self._options.get_option_default(field) is None]
        if missing:
            raise HashiVaultValueError('Authentication method %s requires options %r to be set, but these are missing: %r' % (self.NAME, field_names, missing))

    def warn(self, message):
        self._warner(message)

    def deprecate(self, message, version=None, date=None, collection_name=None):
        self._deprecator(message, version=version, date=date, collection_name=collection_name)