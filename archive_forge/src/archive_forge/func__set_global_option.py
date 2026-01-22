from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def _set_global_option(username, _config=None):
    if _config is None:
        _config = GlobalStack()
    if username is None:
        _config.remove('launchpad_username')
    else:
        _config.set('launchpad_username', username)