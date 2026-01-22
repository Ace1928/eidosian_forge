from keystoneauth1 import exceptions
from keystoneauth1.loading import base
from keystoneauth1.loading import opts
class BaseIdentityLoader(base.BaseLoader):
    """Base Option handling for identity plugins.

    This class defines options and handling that should be common across all
    plugins that are developed against the OpenStack identity service. It
    provides the options expected by the
    :py:class:`keystoneauth1.identity.BaseIdentityPlugin` class.
    """

    def get_options(self):
        options = super(BaseIdentityLoader, self).get_options()
        options.extend([opts.Opt('auth-url', required=True, help='Authentication URL')])
        return options