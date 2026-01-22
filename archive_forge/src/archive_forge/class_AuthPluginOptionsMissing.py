import inspect
import sys
from magnumclient.i18n import _
class AuthPluginOptionsMissing(AuthorizationFailure):
    """Auth plugin misses some options."""

    def __init__(self, opt_names):
        super(AuthPluginOptionsMissing, self).__init__(_('Authentication failed. Missing options: %s') % ', '.join(opt_names))
        self.opt_names = opt_names