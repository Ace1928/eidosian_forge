from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def _set_auth_user(username, auth=None):
    if auth is None:
        auth = AuthenticationConfig()
    auth.set_credentials('Launchpad', '.launchpad.net', username, 'ssh')