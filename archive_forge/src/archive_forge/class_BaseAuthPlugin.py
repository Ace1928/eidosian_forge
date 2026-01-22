import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
class BaseAuthPlugin(object, metaclass=abc.ABCMeta):
    """Base class for authentication plugins.

    An authentication plugin needs to override at least the authenticate
    method to be a valid plugin.
    """
    auth_system = None
    opt_names = []
    common_opt_names = ['auth_system', 'username', 'password', 'auth_url']

    def __init__(self, auth_system=None, **kwargs):
        self.auth_system = auth_system or self.auth_system
        self.opts = dict(((name, kwargs.get(name)) for name in self.opt_names))

    @staticmethod
    def _parser_add_opt(parser, opt):
        """Add an option to parser in two variants.

        :param opt: option name (with underscores)
        """
        dashed_opt = opt.replace('_', '-')
        env_var = 'OS_%s' % opt.upper()
        arg_default = os.environ.get(env_var, '')
        arg_help = 'Defaults to env[%s].' % env_var
        parser.add_argument('--os-%s' % dashed_opt, metavar='<%s>' % dashed_opt, default=arg_default, help=arg_help)
        parser.add_argument('--os_%s' % opt, metavar='<%s>' % dashed_opt, help=argparse.SUPPRESS)

    @classmethod
    def add_opts(cls, parser):
        """Populate the parser with the options for this plugin."""
        for opt in cls.opt_names:
            if opt not in BaseAuthPlugin.common_opt_names:
                cls._parser_add_opt(parser, opt)

    @classmethod
    def add_common_opts(cls, parser):
        """Add options that are common for several plugins."""
        for opt in cls.common_opt_names:
            cls._parser_add_opt(parser, opt)

    @staticmethod
    def get_opt(opt_name, args):
        """Return option name and value.

        :param opt_name: name of the option, e.g., "username"
        :param args: parsed arguments
        """
        return (opt_name, getattr(args, 'os_%s' % opt_name, None))

    def parse_opts(self, args):
        """Parse the actual auth-system options if any.

        This method is expected to populate the attribute `self.opts` with a
        dict containing the options and values needed to make authentication.
        """
        self.opts.update(dict((self.get_opt(opt_name, args) for opt_name in self.opt_names)))

    def authenticate(self, http_client):
        """Authenticate using plugin defined method.

        The method usually analyses `self.opts` and performs
        a request to authentication server.

        :param http_client: client object that needs authentication
        :type http_client: HTTPClient
        :raises: AuthorizationFailure
        """
        self.sufficient_options()
        self._do_authenticate(http_client)

    @abc.abstractmethod
    def _do_authenticate(self, http_client):
        """Protected method for authentication."""

    def sufficient_options(self):
        """Check if all required options are present.

        :raises: AuthPluginOptionsMissing
        """
        missing = [opt for opt in self.opt_names if not self.opts.get(opt)]
        if missing:
            raise exceptions.AuthPluginOptionsMissing(missing)