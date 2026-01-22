import six
from pycadf import cadftype
from pycadf import utils
class FederatedCredential(Credential):
    identity_provider = cadftype.ValidatorDescriptor(FED_CRED_KEYNAME_IDENTITY_PROVIDER, lambda x: isinstance(x, six.string_types))
    user = cadftype.ValidatorDescriptor(FED_CRED_KEYNAME_USER, lambda x: isinstance(x, six.string_types))
    groups = cadftype.ValidatorDescriptor(FED_CRED_KEYNAME_GROUPS, lambda x: isinstance(x, list))

    def __init__(self, token, type, identity_provider, user, groups):
        super(FederatedCredential, self).__init__(token=token, type=type)
        setattr(self, FED_CRED_KEYNAME_IDENTITY_PROVIDER, identity_provider)
        setattr(self, FED_CRED_KEYNAME_USER, user)
        setattr(self, FED_CRED_KEYNAME_GROUPS, groups)

    def is_valid(self):
        """Validation to ensure Credential required attributes are set."""
        return super(FederatedCredential, self).is_valid() and self._isset(CRED_KEYNAME_TYPE) and self._isset(FED_CRED_KEYNAME_IDENTITY_PROVIDER) and self._isset(FED_CRED_KEYNAME_USER) and self._isset(FED_CRED_KEYNAME_GROUPS)