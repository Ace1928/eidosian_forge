import argparse
import getpass
import glob
import importlib
import itertools
import logging
import os
import pkgutil
import sys
from keystoneauth1.identity.generic import password
from keystoneauth1.identity.generic import token
from keystoneauth1 import loading
from oslo_utils import encodeutils
from oslo_utils import importutils
import pkg_resources
from troveclient.apiclient import exceptions as exc
import troveclient.auth_plugin
from troveclient import client
import troveclient.extension
from troveclient.i18n import _  # noqa
from troveclient import utils
from troveclient.v1 import shell as shell_v1
class OpenStackTroveShell(object):

    def get_base_parser(self, argv):
        parser = TroveClientArgumentParser(prog='trove', description=__doc__.strip(), epilog=_('See "trove help COMMAND" for help on a specific command.'), add_help=False, formatter_class=OpenStackHelpFormatter)
        parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--version', action='version', version=troveclient.__version__, help=_("Show program's version number and exit."))
        parser.add_argument('--debug', action='store_true', default=utils.env('TROVECLIENT_DEBUG', default=False), help=_('Print debugging output.'))
        parser.add_argument('--os-auth-system', metavar='<auth-system>', default=utils.env('OS_AUTH_SYSTEM'), help=argparse.SUPPRESS)
        parser.add_argument('--os_auth_system', help=argparse.SUPPRESS)
        parser.add_argument('--service-type', metavar='<service-type>', default=utils.env('OS_SERVICE_TYPE', 'TROVE_SERVICE_TYPE'), help=_('Defaults to database for most actions.'))
        parser.add_argument('--service_type', help=argparse.SUPPRESS)
        parser.add_argument('--service-name', metavar='<service-name>', default=utils.env('TROVE_SERVICE_NAME'), help=_('Defaults to env[TROVE_SERVICE_NAME].'))
        parser.add_argument('--service_name', help=argparse.SUPPRESS)
        parser.add_argument('--bypass-url', metavar='<bypass-url>', default=utils.env('TROVE_BYPASS_URL'), help=_('Defaults to env[TROVE_BYPASS_URL].'))
        parser.add_argument('--bypass_url', help=argparse.SUPPRESS)
        parser.add_argument('--database-service-name', metavar='<database-service-name>', default=utils.env('TROVE_DATABASE_SERVICE_NAME'), help=_('Defaults to env[TROVE_DATABASE_SERVICE_NAME].'))
        parser.add_argument('--database_service_name', help=argparse.SUPPRESS)
        default_trove_endpoint_type = utils.env('OS_ENDPOINT_TYPE', default=DEFAULT_TROVE_ENDPOINT_TYPE)
        parser.add_argument('--endpoint-type', metavar='<endpoint-type>', default=utils.env('TROVE_ENDPOINT_TYPE', default=default_trove_endpoint_type), help=_('Defaults to env[TROVE_ENDPOINT_TYPE] or env[OS_ENDPOINT_TYPE] or %(DEFAULT_TROVE_ENDPOINT_TYPE)s.') % {'DEFAULT_TROVE_ENDPOINT_TYPE': DEFAULT_TROVE_ENDPOINT_TYPE})
        parser.add_argument('--endpoint_type', help=argparse.SUPPRESS)
        parser.add_argument('--os-database-api-version', metavar='<database-api-ver>', default=utils.env('OS_DATABASE_API_VERSION', default=DEFAULT_OS_DATABASE_API_VERSION), help=_('Accepts 1, defaults to env[OS_DATABASE_API_VERSION].'))
        parser.add_argument('--os_database_api_version', help=argparse.SUPPRESS)
        parser.add_argument('--retries', metavar='<retries>', type=int, default=0, help=_('Number of retries.'))
        parser.add_argument('--json', '--os-json-output', dest='json', action='store_true', default=utils.env('OS_JSON_OUTPUT', default=False), help=_('Output JSON instead of prettyprint. Defaults to env[OS_JSON_OUTPUT].'))
        if osprofiler_profiler:
            parser.add_argument('--profile', metavar='HMAC_KEY', default=utils.env('OS_PROFILE_HMACKEY', default=None), help=_('HMAC key used to encrypt context data when profiling the performance of an operation. This key should be set to one of the HMAC keys configured in Trove (they are found in configuration files, typically in /etc/trove). Without the key, profiling will not be triggered even if it is enabled on the server side. Defaults to env[OS_PROFILE_HMACKEY].'))
        self._append_global_identity_args(parser, argv)
        troveclient.auth_plugin.load_auth_system_opts(parser)
        return parser

    def _append_global_identity_args(self, parser, argv):
        loading.register_session_argparse_arguments(parser)
        default_auth_plugin = 'password'
        if 'os-token' in argv:
            default_auth_plugin = 'token'
        loading.register_auth_argparse_arguments(parser, argv, default=default_auth_plugin)
        parser.set_defaults(insecure=utils.env('TROVECLIENT_INSECURE', default=False))
        parser.set_defaults(os_auth_url=utils.env('OS_AUTH_URL'))
        parser.set_defaults(os_project_name=utils.env('OS_PROJECT_NAME', 'OS_TENANT_NAME'))
        parser.set_defaults(os_project_id=utils.env('OS_PROJECT_ID', 'OS_TENANT_ID'))
        parser.add_argument('--os_tenant_name', help=argparse.SUPPRESS)
        parser.add_argument('--os_tenant_id', help=argparse.SUPPRESS)
        parser.add_argument('--os-auth-token', default=utils.env('OS_AUTH_TOKEN'), help=argparse.SUPPRESS)
        parser.add_argument('--os-region-name', metavar='<region-name>', default=utils.env('OS_REGION_NAME'), help=_('Specify the region to use. Defaults to env[OS_REGION_NAME].'))
        parser.add_argument('--os_region_name', help=argparse.SUPPRESS)

    def get_subcommand_parser(self, version, argv):
        parser = self.get_base_parser(argv)
        self.subcommands = {}
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        try:
            actions_module = {'1.0': shell_v1}[version]
        except KeyError:
            actions_module = shell_v1
        self._find_actions(subparsers, actions_module)
        self._find_actions(subparsers, self)
        for extension in self.extensions:
            self._find_actions(subparsers, extension.module)
        self._add_bash_completion_subparser(subparsers)
        return parser

    def _discover_extensions(self, version):
        extensions = []
        for name, module in itertools.chain(self._discover_via_python_path(), self._discover_via_contrib_path(version), self._discover_via_entry_points()):
            extension = troveclient.extension.Extension(name, module)
            extensions.append(extension)
        return extensions

    def _discover_via_python_path(self):
        for module_loader, name, _ispkg in pkgutil.iter_modules():
            if name.endswith('_python_troveclient_ext'):
                if not hasattr(module_loader, 'load_module'):
                    module_loader = module_loader.find_module(name)
                module = module_loader.load_module(name)
                if hasattr(module, 'extension_name'):
                    name = module.extension_name
                yield (name, module)

    def _load_module(self, name, path):
        module_spec = importlib.spec_from_file_location(name, path)
        module = importlib.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def _discover_via_contrib_path(self, version):
        module_path = os.path.dirname(os.path.abspath(__file__))
        version_str = 'v%s' % version.replace('.', '_')
        version_pkg = 'v1' if version_str == 'v1_0' else version_str
        ext_path = os.path.join(module_path, version_pkg, 'contrib')
        ext_glob = os.path.join(ext_path, '*.py')
        for ext_path in glob.iglob(ext_glob):
            name = os.path.basename(ext_path)[:-3]
            if name == '__init__':
                continue
            module = self._load_module(name, ext_path)
            yield (name, module)

    def _discover_via_entry_points(self):
        for ep in pkg_resources.iter_entry_points('troveclient.extension'):
            name = ep.name
            module = ep.load()
            yield (name, module)

    def _add_bash_completion_subparser(self, subparsers):
        subparser = subparsers.add_parser('bash_completion', add_help=False, formatter_class=OpenStackHelpFormatter)
        self.subcommands['bash_completion'] = subparser
        subparser.set_defaults(func=self.do_bash_completion)

    def _find_actions(self, subparsers, actions_module):
        for attr in (a for a in dir(actions_module) if a.startswith('do_')):
            command = attr[3:].replace('_', '-')
            callback = getattr(actions_module, attr)
            desc = callback.__doc__ or ''
            help = desc.strip().split('\n')[0]
            arguments = getattr(callback, 'arguments', [])
            subparser = subparsers.add_parser(command, help=help, description=desc, add_help=False, formatter_class=OpenStackHelpFormatter)
            subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
            self.subcommands[command] = subparser
            for args, kwargs in arguments:
                subparser.add_argument(*args, **kwargs)
            subparser.set_defaults(func=callback)

    def setup_debugging(self, debug):
        if not debug:
            return
        streamformat = '%(levelname)s (%(module)s:%(lineno)d) %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=streamformat)

    def main(self, argv):
        parser = self.get_base_parser(argv)
        options, args = parser.parse_known_args(argv)
        self.setup_debugging(options.debug)
        self.options = options
        troveclient.auth_plugin.discover_auth_systems()
        self.extensions = self._discover_extensions(options.os_database_api_version)
        self._run_extension_hooks('__pre_parse_args__')
        subcommand_parser = self.get_subcommand_parser(options.os_database_api_version, argv)
        self.parser = subcommand_parser
        if options.help or not argv:
            subcommand_parser.print_help()
            return 0
        args = subcommand_parser.parse_args(argv)
        self._run_extension_hooks('__post_parse_args__', args)
        if args.func == self.do_help:
            self.do_help(args)
            return 0
        elif args.func == self.do_bash_completion:
            self.do_bash_completion(args)
            return 0
        os_username = args.os_username
        os_password = args.os_password
        os_project_name = getattr(args, 'os_project_name', getattr(args, 'os_tenant_name', None))
        os_auth_url = args.os_auth_url
        os_region_name = args.os_region_name
        os_project_id = getattr(args, 'os_project_id', getattr(args, 'os_tenant_id', None))
        os_auth_system = args.os_auth_system
        if 'v2.0' not in os_auth_url:
            if not args.os_project_domain_id and (not args.os_project_domain_name):
                setattr(args, 'os_project_domain_id', 'default')
            if not args.os_user_domain_id and (not args.os_user_domain_name):
                setattr(args, 'os_user_domain_id', 'default')
        endpoint_type = args.endpoint_type
        insecure = args.insecure
        service_type = args.service_type
        service_name = args.service_name
        database_service_name = args.database_service_name
        cacert = args.os_cacert
        bypass_url = args.bypass_url
        if os_auth_system and os_auth_system != 'keystone':
            auth_plugin = troveclient.auth_plugin.load_plugin(os_auth_system)
        else:
            auth_plugin = None
        if not endpoint_type:
            endpoint_type = DEFAULT_TROVE_ENDPOINT_TYPE
        if not service_type:
            service_type = DEFAULT_TROVE_SERVICE_TYPE
            service_type = utils.get_service_type(args.func) or service_type
        if not utils.isunauthenticated(args.func):
            if auth_plugin:
                auth_plugin.parse_opts(args)
            if not auth_plugin or not auth_plugin.opts:
                if not os_username:
                    raise exc.CommandError(_('You must provide a username via either --os-username or env[OS_USERNAME]'))
            if not os_password:
                os_password = getpass.getpass()
            if not os_auth_url:
                if os_auth_system and os_auth_system != 'keystone':
                    os_auth_url = auth_plugin.get_auth_url()
        project_info_provided = self.options.os_project_name or self.options.os_project_id
        if not project_info_provided:
            raise exc.CommandError(_('You must provide a project_id or project_name (with project_domain_name or project_domain_id) via   --os-project-id (env[OS_PROJECT_ID])  --os-project-name (env[OS_PROJECT_NAME]),  --os-project-domain-id (env[OS_PROJECT_DOMAIN_ID])  --os-project-domain-name (env[OS_PROJECT_DOMAIN_NAME])'))
        if not os_auth_url:
            raise exc.CommandError(_('You must provide an auth url via either --os-auth-url or env[OS_AUTH_URL] or specify an auth_system which defines a default url with --os-auth-system or env[OS_AUTH_SYSTEM]'))
        use_session = True
        if auth_plugin or bypass_url:
            use_session = False
        ks_session = None
        keystone_auth = None
        if use_session:
            project_id = args.os_project_id or args.os_tenant_id
            project_name = args.os_project_name or args.os_tenant_name
            ks_session = loading.load_session_from_argparse_arguments(args)
            keystone_auth = self._get_keystone_auth(ks_session, args.os_auth_url, username=args.os_username, user_id=args.os_user_id, user_domain_id=args.os_user_domain_id, user_domain_name=args.os_user_domain_name, password=args.os_password, auth_token=args.os_auth_token, project_id=project_id, project_name=project_name, project_domain_id=args.os_project_domain_id, project_domain_name=args.os_project_domain_name)
        profile = osprofiler_profiler and options.profile
        if profile:
            osprofiler_profiler.init(options.profile)
        self.cs = client.Client(options.os_database_api_version, os_username, os_password, os_project_name, os_auth_url, insecure, region_name=os_region_name, tenant_id=os_project_id, endpoint_type=endpoint_type, extensions=self.extensions, service_type=service_type, service_name=service_name, database_service_name=database_service_name, retries=options.retries, http_log_debug=args.debug, cacert=cacert, bypass_url=bypass_url, auth_system=os_auth_system, auth_plugin=auth_plugin, session=ks_session, auth=keystone_auth)
        try:
            if not utils.isunauthenticated(args.func):
                if not use_session:
                    self.cs.authenticate()
        except exc.Unauthorized:
            raise exc.CommandError(_('Invalid OpenStack Trove credentials.'))
        except exc.AuthorizationFailure:
            raise exc.CommandError(_('Unable to authorize user'))
        endpoint_api_version = self.cs.get_database_api_version_from_endpoint()
        if endpoint_api_version != options.os_database_api_version:
            msg = _('Database API version is set to %(db_ver)s but you are accessing a %(ep_ver)s endpoint. Change its value via either --os-database-api-version or env[OS_DATABASE_API_VERSION]') % {'db_ver': options.os_database_api_version, 'ep_ver': endpoint_api_version}
            raise exc.UnsupportedVersion(msg)
        if args.json:
            utils.json_output = True
        else:
            utils.json_output = False
        try:
            args.func(self.cs, args)
        finally:
            if profile:
                trace_id = osprofiler_profiler.get().get_base_id()
                print(_('Trace ID: %(trace_id)s') % {'trace_id': trace_id})
                print(_('To display the trace, use the following command:\nosprofiler trace show --html %(trace_id)s') % {'trace_id': trace_id})

    def _run_extension_hooks(self, hook_type, *args, **kwargs):
        """Run hooks for all registered extensions."""
        for extension in self.extensions:
            extension.run_hooks(hook_type, *args, **kwargs)

    def do_bash_completion(self, args):
        """Prints arguments for bash_completion.

        Prints all of the commands and options to stdout so that the
        trove.bash_completion script doesn't have to hard code them.
        """
        commands = set()
        options = set()
        for sc_str, sc in list(self.subcommands.items()):
            commands.add(sc_str)
            for option in list(sc._optionals._option_string_actions.keys()):
                options.add(option)
        commands.remove('bash-completion')
        commands.remove('bash_completion')
        print(' '.join(commands | options))

    @utils.arg('command', metavar='<subcommand>', nargs='?', help=_('Display help for <subcommand>.'))
    def do_help(self, args):
        """Displays help about this program or one of its subcommands."""
        if args.command:
            if args.command in self.subcommands:
                self.subcommands[args.command].print_help()
            else:
                raise exc.CommandError(_("'%s' is not a valid subcommand") % args.command)
        else:
            self.parser.print_help()

    def _get_keystone_auth(self, session, auth_url, **kwargs):
        auth_token = kwargs.pop('auth_token', None)
        if auth_token:
            return token.Token(auth_url, auth_token, **kwargs)
        else:
            return password.Password(auth_url, username=kwargs.pop('username'), user_id=kwargs.pop('user_id'), password=kwargs.pop('password'), user_domain_id=kwargs.pop('user_domain_id'), user_domain_name=kwargs.pop('user_domain_name'), **kwargs)