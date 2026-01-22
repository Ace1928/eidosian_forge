import argparse
import copy
import getpass
import hashlib
import json
import logging
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import importutils
import urllib.parse
import glanceclient
from glanceclient._i18n import _
from glanceclient.common import utils
from glanceclient import exc
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
class OpenStackImagesShell(object):

    def _append_global_identity_args(self, parser, argv):
        parser.set_defaults(os_auth_url=utils.env('OS_AUTH_URL'))
        parser.set_defaults(os_project_name=utils.env('OS_PROJECT_NAME', 'OS_TENANT_NAME'))
        parser.set_defaults(os_project_id=utils.env('OS_PROJECT_ID', 'OS_TENANT_ID'))
        parser.add_argument('--os_tenant_id', help=argparse.SUPPRESS)
        parser.add_argument('--os_tenant_name', help=argparse.SUPPRESS)
        parser.add_argument('--os-region-name', default=utils.env('OS_REGION_NAME'), help='Defaults to env[OS_REGION_NAME].')
        parser.add_argument('--os_region_name', help=argparse.SUPPRESS)
        parser.add_argument('--os-auth-token', default=utils.env('OS_AUTH_TOKEN'), help='Defaults to env[OS_AUTH_TOKEN].')
        parser.add_argument('--os_auth_token', help=argparse.SUPPRESS)
        parser.add_argument('--os-service-type', default=utils.env('OS_SERVICE_TYPE'), help='Defaults to env[OS_SERVICE_TYPE].')
        parser.add_argument('--os_service_type', help=argparse.SUPPRESS)
        parser.add_argument('--os-endpoint-type', default=utils.env('OS_ENDPOINT_TYPE'), help='Defaults to env[OS_ENDPOINT_TYPE].')
        parser.add_argument('--os_endpoint_type', help=argparse.SUPPRESS)
        loading.register_session_argparse_arguments(parser)
        default_auth_plugin = 'password'
        token_opts = ['os-token', 'os-auth-token', 'os_auth-token']
        if argv and any((i in token_opts for i in argv)):
            default_auth_plugin = 'token'
        loading.register_auth_argparse_arguments(parser, argv, default=default_auth_plugin)

    def get_base_parser(self, argv):
        parser = argparse.ArgumentParser(prog='glance', description=__doc__.strip(), epilog='See "glance help COMMAND" for help on a specific command.', add_help=False, formatter_class=HelpFormatter)
        parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--version', action='version', version=glanceclient.__version__)
        parser.add_argument('-d', '--debug', default=bool(utils.env('GLANCECLIENT_DEBUG')), action='store_true', help='Defaults to env[GLANCECLIENT_DEBUG].')
        parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Print more verbose output.')
        parser.add_argument('--get-schema', default=False, action='store_true', dest='get_schema', help='Ignores cached copy and forces retrieval of schema that generates portions of the help text. Ignored with API version 1.')
        parser.add_argument('-f', '--force', dest='force', default=False, action='store_true', help='Prevent select actions from requesting user confirmation.')
        parser.add_argument('--os-image-url', default=utils.env('OS_IMAGE_URL'), help='Defaults to env[OS_IMAGE_URL]. If the provided image url contains a version number and `--os-image-api-version` is omitted the version of the URL will be picked as the image api version to use.')
        parser.add_argument('--os_image_url', help=argparse.SUPPRESS)
        parser.add_argument('--os-image-api-version', default=utils.env('OS_IMAGE_API_VERSION', default=None), help='Defaults to env[OS_IMAGE_API_VERSION] or 2.')
        parser.add_argument('--os_image_api_version', help=argparse.SUPPRESS)
        parser.set_defaults(func=self.do_help)
        if osprofiler_profiler:
            parser.add_argument('--profile', metavar='HMAC_KEY', default=utils.env('OS_PROFILE'), help='HMAC key to use for encrypting context data for performance profiling of operation. This key should be the value of HMAC key configured in osprofiler middleware in glance, it is specified in glance configuration file at /etc/glance/glance-api.conf and /etc/glance/glance-registry.conf. Without key the profiling will not be triggered even if osprofiler is enabled on server side. Defaults to env[OS_PROFILE].')
        self._append_global_identity_args(parser, argv)
        return parser

    def get_subcommand_parser(self, version, argv=None):
        parser = self.get_base_parser(argv)
        self.subcommands = {}
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        submodule = importutils.import_versioned_module('glanceclient', version, 'shell')
        self._find_actions(subparsers, submodule)
        self._find_actions(subparsers, self)
        self._add_bash_completion_subparser(subparsers)
        return parser

    def _find_actions(self, subparsers, actions_module):
        for attr in (a for a in dir(actions_module) if a.startswith('do_')):
            command = attr[3:].replace('_', '-')
            callback = getattr(actions_module, attr)
            desc = callback.__doc__ or ''
            help = desc.strip().split('\n')[0]
            arguments = getattr(callback, 'arguments', [])
            subparser = subparsers.add_parser(command, help=help, description=desc, add_help=False, formatter_class=HelpFormatter)
            subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
            self.subcommands[command] = subparser
            required_args = subparser.add_argument_group('Required arguments')
            for args, kwargs in arguments:
                if kwargs.get('required', False):
                    required_args.add_argument(*args, **kwargs)
                else:
                    subparser.add_argument(*args, **kwargs)
            subparser.set_defaults(func=callback)

    def _add_bash_completion_subparser(self, subparsers):
        subparser = subparsers.add_parser('bash_completion', add_help=False, formatter_class=HelpFormatter)
        self.subcommands['bash_completion'] = subparser
        subparser.set_defaults(func=self.do_bash_completion)

    def _get_image_url(self, args):
        """Translate the available url-related options into a single string.

        Return the endpoint that should be used to talk to Glance if a
        clear decision can be made. Otherwise, return None.
        """
        if args.os_image_url:
            return args.os_image_url
        else:
            return None

    def _discover_auth_versions(self, session, auth_url):
        v2_auth_url = None
        v3_auth_url = None
        try:
            ks_discover = discover.Discover(session=session, url=auth_url)
            v2_auth_url = ks_discover.url_for('2.0')
            v3_auth_url = ks_discover.url_for('3.0')
        except ks_exc.ClientException as e:
            url_parts = urllib.parse.urlparse(auth_url)
            scheme, netloc, path, params, query, fragment = url_parts
            path = path.lower()
            if path.startswith('/v3'):
                v3_auth_url = auth_url
            elif path.startswith('/v2'):
                v2_auth_url = auth_url
            else:
                msg = 'Unable to determine the Keystone version to authenticate with using the given auth_url. Identity service may not support API version discovery. Please provide a versioned auth_url instead. error=%s' % e
                raise exc.CommandError(msg)
        return (v2_auth_url, v3_auth_url)

    def _get_keystone_auth_plugin(self, ks_session, **kwargs):
        auth_url = kwargs.pop('auth_url', None)
        v2_auth_url, v3_auth_url = self._discover_auth_versions(session=ks_session, auth_url=auth_url)
        user_id = kwargs.pop('user_id', None)
        username = kwargs.pop('username', None)
        password = kwargs.pop('password', None)
        user_domain_name = kwargs.pop('user_domain_name', None)
        user_domain_id = kwargs.pop('user_domain_id', None)
        project_id = kwargs.pop('project_id', None) or kwargs.pop('tenant_id', None)
        project_name = kwargs.pop('project_name', None) or kwargs.pop('tenant_name', None)
        project_domain_id = kwargs.pop('project_domain_id', None)
        project_domain_name = kwargs.pop('project_domain_name', None)
        auth = None
        use_domain = user_domain_id or user_domain_name or project_domain_id or project_domain_name
        use_v3 = v3_auth_url and (use_domain or not v2_auth_url)
        use_v2 = v2_auth_url and (not use_domain)
        if use_v3:
            auth = v3_auth.Password(v3_auth_url, user_id=user_id, username=username, password=password, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
        elif use_v2:
            auth = v2_auth.Password(v2_auth_url, username, password, tenant_id=project_id, tenant_name=project_name)
        else:
            exc.CommandError('Credential and auth_url mismatch. The given auth_url is using Keystone V2 endpoint, which may not able to handle Keystone V3 credentials. Please provide a correct Keystone V3 auth_url.')
        return auth

    def _get_kwargs_to_create_auth_plugin(self, args):
        if not args.os_username:
            raise exc.CommandError(_('You must provide a username via either --os-username or env[OS_USERNAME]'))
        if not args.os_password:
            if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
                try:
                    args.os_password = getpass.getpass('OS Password: ')
                except EOFError:
                    pass
            if not args.os_password:
                raise exc.CommandError(_('You must provide a password via either --os-password, env[OS_PASSWORD], or prompted response'))
        os_project_name = getattr(args, 'os_project_name', getattr(args, 'os_tenant_name', None))
        os_project_id = getattr(args, 'os_project_id', getattr(args, 'os_tenant_id', None))
        if not any([os_project_name, os_project_id]):
            raise exc.CommandError(_('You must provide a project_id or project_name (with project_domain_name or project_domain_id) via   --os-project-id (env[OS_PROJECT_ID])  --os-project-name (env[OS_PROJECT_NAME]),  --os-project-domain-id (env[OS_PROJECT_DOMAIN_ID])  --os-project-domain-name (env[OS_PROJECT_DOMAIN_NAME])'))
        if not args.os_auth_url:
            raise exc.CommandError(_('You must provide an auth url via either --os-auth-url or via env[OS_AUTH_URL]'))
        kwargs = {'auth_url': args.os_auth_url, 'username': args.os_username, 'user_id': args.os_user_id, 'user_domain_id': args.os_user_domain_id, 'user_domain_name': args.os_user_domain_name, 'password': args.os_password, 'tenant_name': args.os_tenant_name, 'tenant_id': args.os_tenant_id, 'project_name': args.os_project_name, 'project_id': args.os_project_id, 'project_domain_name': args.os_project_domain_name, 'project_domain_id': args.os_project_domain_id}
        return kwargs

    def _get_versioned_client(self, api_version, args):
        endpoint = self._get_image_url(args)
        auth_token = args.os_auth_token
        if endpoint and auth_token:
            kwargs = {'token': auth_token, 'insecure': args.insecure, 'timeout': args.timeout, 'cacert': args.os_cacert, 'cert': args.os_cert, 'key': args.os_key}
        else:
            ks_session = loading.load_session_from_argparse_arguments(args)
            auth_plugin_kwargs = self._get_kwargs_to_create_auth_plugin(args)
            ks_session.auth = self._get_keystone_auth_plugin(ks_session=ks_session, **auth_plugin_kwargs)
            kwargs = {'session': ks_session}
            if endpoint is None:
                endpoint_type = args.os_endpoint_type or 'public'
                service_type = args.os_service_type or 'image'
                endpoint = ks_session.get_endpoint(service_type=service_type, interface=endpoint_type, region_name=args.os_region_name)
        return glanceclient.Client(api_version, endpoint, **kwargs)

    def _cache_schemas(self, options, client, home_dir='~/.glanceclient'):
        homedir = os.path.expanduser(home_dir)
        path_prefix = homedir
        if options.os_auth_url:
            hash_host = hashlib.sha1(options.os_auth_url.encode('utf-8'))
            path_prefix = os.path.join(path_prefix, hash_host.hexdigest())
        if not os.path.exists(path_prefix):
            try:
                os.makedirs(path_prefix)
            except OSError as e:
                msg = '%s' % e
                print(encodeutils.safe_decode(msg), file=sys.stderr)
        resources = ['image', 'metadefs/namespace', 'metadefs/resource_type']
        schema_file_paths = [os.path.join(path_prefix, x + '_schema.json') for x in ['image', 'namespace', 'resource_type']]
        failed_download_schema = 0
        for resource, schema_file_path in zip(resources, schema_file_paths):
            if not os.path.exists(schema_file_path) or options.get_schema:
                try:
                    schema = client.schemas.get(resource)
                    with open(schema_file_path, 'w') as f:
                        f.write(json.dumps(schema.raw()))
                except exc.Unauthorized:
                    raise exc.CommandError('Invalid OpenStack Identity credentials.')
                except Exception:
                    failed_download_schema += 1
                    pass
        return failed_download_schema >= len(resources)

    def main(self, argv):

        def _get_subparser(api_version):
            try:
                return self.get_subcommand_parser(api_version, argv)
            except ImportError as e:
                if not str(e):
                    raise ImportError('Unable to import module. Re-run with --debug for more info.')
                raise
        base_argv = copy.deepcopy(argv)
        parser = self.get_base_parser(argv)
        options, args = parser.parse_known_args(base_argv)
        try:
            endpoint = self._get_image_url(options)
            endpoint, url_version = utils.strip_version(endpoint)
        except ValueError:
            url_version = None
        try:
            api_version = int(options.os_image_api_version or url_version or 2)
            if api_version not in SUPPORTED_VERSIONS:
                raise ValueError
        except ValueError:
            msg = 'Invalid API version parameter. Supported values are %s' % SUPPORTED_VERSIONS
            utils.exit(msg=msg)
        if options.help or not argv:
            parser = _get_subparser(api_version)
            self.do_help(options, parser=parser)
            return 0
        sub_parser = _get_subparser(api_version)
        args = sub_parser.parse_args(argv)
        if args.func == self.do_help:
            self.do_help(args, parser=sub_parser)
            return 0
        elif args.func == self.do_bash_completion:
            self.do_bash_completion(args)
            return 0
        if not options.os_image_api_version and api_version == 2:
            switch_version = True
            client = self._get_versioned_client('2', args)
            resp, body = client.http_client.get('/versions')
            for version in body['versions']:
                if version['id'].startswith('v2'):
                    switch_version = self._cache_schemas(options, client)
                    break
            if switch_version:
                print('WARNING: The client is falling back to v1 because the accessing to v2 failed. This behavior will be removed in future versions', file=sys.stderr)
                api_version = 1
        sub_parser = _get_subparser(api_version)
        args = sub_parser.parse_args(argv)
        if not args.os_password and options.os_password:
            args.os_password = options.os_password
        if args.debug:
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger('iso8601').setLevel(logging.WARNING)
        LOG = logging.getLogger('glanceclient')
        LOG.addHandler(logging.StreamHandler())
        LOG.setLevel(logging.DEBUG if args.debug else logging.INFO)
        profile = osprofiler_profiler and options.profile
        if profile:
            osprofiler_profiler.init(options.profile)
        client = self._get_versioned_client(api_version, args)
        try:
            args.func(client, args)
        except exc.Unauthorized:
            raise exc.CommandError('Invalid OpenStack Identity credentials.')
        finally:
            if profile:
                trace_id = osprofiler_profiler.get().get_base_id()
                print('Profiling trace ID: %s' % trace_id)
                print('To display trace use next command:\nosprofiler trace show --html %s ' % trace_id)

    @utils.arg('command', metavar='<subcommand>', nargs='?', help='Display help for <subcommand>.')
    def do_help(self, args, parser):
        """Display help about this program or one of its subcommands."""
        command = getattr(args, 'command', '')
        if command:
            if args.command in self.subcommands:
                self.subcommands[args.command].print_help()
            else:
                raise exc.CommandError("'%s' is not a valid subcommand" % args.command)
        else:
            parser.print_help()
        if not args.os_image_api_version or args.os_image_api_version == '2':
            try:
                if command is None:
                    print('\nRun `glance --os-image-api-version 1 help` for v1 help')
                else:
                    self.get_subcommand_parser(1)
                    if command in self.subcommands:
                        command = ' ' + command
                        print('\nRun `glance --os-image-api-version 1 help%s` for v1 help' % (command or ''))
            except ImportError:
                pass

    def do_bash_completion(self, _args):
        """Prints arguments for bash_completion.

        Prints all of the commands and options to stdout so that the
        glance.bash_completion script doesn't have to hard code them.
        """
        commands = set()
        options = set()
        for sc_str, sc in self.subcommands.items():
            commands.add(sc_str)
            for option in sc._optionals._option_string_actions.keys():
                options.add(option)
        commands.remove('bash_completion')
        commands.remove('bash-completion')
        print(' '.join(commands | options))