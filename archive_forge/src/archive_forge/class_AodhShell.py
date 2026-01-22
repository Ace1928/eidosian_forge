import logging
import os
import sys
import warnings
from cliff import app
from cliff import commandmanager
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from aodhclient import __version__
from aodhclient import client
from aodhclient import noauth
from aodhclient.v2 import alarm_cli
from aodhclient.v2 import alarm_history_cli
from aodhclient.v2 import capabilities_cli
class AodhShell(app.App):

    def __init__(self):
        super(AodhShell, self).__init__(description='Aodh command line client', version=__version__, command_manager=AodhCommandManager('aodhclient'), deferred_help=True)
        self._client = None

    def build_option_parser(self, description, version):
        """Return an argparse option parser for this application.

        Subclasses may override this method to extend
        the parser with more global options.

        :param description: full description of the application
        :paramtype description: str
        :param version: version number for the application
        :paramtype version: str
        """
        parser = super(AodhShell, self).build_option_parser(description, version, argparse_kwargs={'allow_abbrev': False})
        parser.add_argument('--os-region-name', metavar='<auth-region-name>', dest='region_name', default=os.environ.get('OS_REGION_NAME'), help='Authentication region name (Env: OS_REGION_NAME)')
        parser.add_argument('--os-interface', metavar='<interface>', dest='interface', choices=['admin', 'public', 'internal'], default=os.environ.get('OS_INTERFACE'), help='Select an interface type. Valid interface types: [admin, public, internal]. (Env: OS_INTERFACE)')
        parser.add_argument('--aodh-api-version', default=os.environ.get('AODH_API_VERSION', '2'), help='Defaults to env[AODH_API_VERSION] or 2.')
        loading.register_session_argparse_arguments(parser=parser)
        plugin = loading.register_auth_argparse_arguments(parser=parser, argv=sys.argv, default='password')
        if not isinstance(plugin, noauth.AodhNoAuthLoader):
            parser.add_argument('--aodh-endpoint', metavar='<endpoint>', dest='endpoint', default=os.environ.get('AODH_ENDPOINT'), help='Aodh endpoint (Env: AODH_ENDPOINT)')
        return parser

    @property
    def client(self):
        if self._client is None:
            if hasattr(self.options, 'endpoint'):
                endpoint_override = self.options.endpoint
            else:
                endpoint_override = None
            auth_plugin = loading.load_auth_from_argparse_arguments(self.options)
            session = loading.load_session_from_argparse_arguments(self.options, auth=auth_plugin)
            self._client = client.Client(self.options.aodh_api_version, session=session, interface=self.options.interface, region_name=self.options.region_name, endpoint_override=endpoint_override)
        return self._client

    def clean_up(self, cmd, result, err):
        if isinstance(err, exceptions.HttpError) and err.details:
            print(err.details, file=sys.stderr)

    def configure_logging(self):
        if self.options.debug:
            self.options.verbose_level = 3
        super(AodhShell, self).configure_logging()
        root_logger = logging.getLogger('')
        if self.options.verbose_level == 0:
            root_logger.setLevel(logging.ERROR)
            warnings.simplefilter('ignore')
        elif self.options.verbose_level == 1:
            root_logger.setLevel(logging.WARNING)
            warnings.simplefilter('ignore')
        elif self.options.verbose_level == 2:
            root_logger.setLevel(logging.INFO)
            warnings.simplefilter('once')
        elif self.options.verbose_level >= 3:
            root_logger.setLevel(logging.DEBUG)
        requests_log = logging.getLogger('requests')
        cliff_log = logging.getLogger('cliff')
        stevedore_log = logging.getLogger('stevedore')
        iso8601_log = logging.getLogger('iso8601')
        cliff_log.setLevel(logging.ERROR)
        stevedore_log.setLevel(logging.ERROR)
        iso8601_log.setLevel(logging.ERROR)
        if self.options.debug:
            requests_log.setLevel(logging.DEBUG)
        else:
            requests_log.setLevel(logging.ERROR)