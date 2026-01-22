from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def _get_auth_user(auth=None):
    if auth is None:
        auth = AuthenticationConfig()
    username = auth.get_user('ssh', '.launchpad.net')
    return username