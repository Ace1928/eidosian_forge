from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
class ControllerModule(AnsibleModule):
    url = None
    AUTH_ARGSPEC = dict(controller_host=dict(required=False, aliases=['tower_host'], fallback=(env_fallback, ['CONTROLLER_HOST', 'TOWER_HOST'])), controller_username=dict(required=False, aliases=['tower_username'], fallback=(env_fallback, ['CONTROLLER_USERNAME', 'TOWER_USERNAME'])), controller_password=dict(no_log=True, aliases=['tower_password'], required=False, fallback=(env_fallback, ['CONTROLLER_PASSWORD', 'TOWER_PASSWORD'])), validate_certs=dict(type='bool', aliases=['tower_verify_ssl'], required=False, fallback=(env_fallback, ['CONTROLLER_VERIFY_SSL', 'TOWER_VERIFY_SSL'])), request_timeout=dict(type='float', required=False, fallback=(env_fallback, ['CONTROLLER_REQUEST_TIMEOUT'])), controller_oauthtoken=dict(type='raw', no_log=True, aliases=['tower_oauthtoken'], required=False, fallback=(env_fallback, ['CONTROLLER_OAUTH_TOKEN', 'TOWER_OAUTH_TOKEN'])), controller_config_file=dict(type='path', aliases=['tower_config_file'], required=False, default=None))
    ordered_associations = ['instance_groups', 'galaxy_credentials']
    short_params = {'host': 'controller_host', 'username': 'controller_username', 'password': 'controller_password', 'verify_ssl': 'validate_certs', 'request_timeout': 'request_timeout', 'oauth_token': 'controller_oauthtoken'}
    host = '127.0.0.1'
    username = None
    password = None
    verify_ssl = True
    request_timeout = 10
    oauth_token = None
    oauth_token_id = None
    authenticated = False
    config_name = 'tower_cli.cfg'
    version_checked = False
    error_callback = None
    warn_callback = None

    def __init__(self, argument_spec=None, direct_params=None, error_callback=None, warn_callback=None, **kwargs):
        full_argspec = {}
        full_argspec.update(ControllerModule.AUTH_ARGSPEC)
        full_argspec.update(argument_spec)
        kwargs['supports_check_mode'] = True
        self.error_callback = error_callback
        self.warn_callback = warn_callback
        self.json_output = {'changed': False}
        if direct_params is not None:
            self.params = direct_params
        else:
            super().__init__(argument_spec=full_argspec, **kwargs)
        self.load_config_files()
        for short_param, long_param in self.short_params.items():
            direct_value = self.params.get(long_param)
            if direct_value is not None:
                setattr(self, short_param, direct_value)
        if self.params.get('controller_oauthtoken'):
            token_param = self.params.get('controller_oauthtoken')
            if type(token_param) is dict:
                if 'token' in token_param:
                    self.oauth_token = self.params.get('controller_oauthtoken')['token']
                else:
                    self.fail_json(msg='The provided dict in controller_oauthtoken did not properly contain the token entry')
            elif isinstance(token_param, string_types):
                self.oauth_token = self.params.get('controller_oauthtoken')
            else:
                error_msg = 'The provided controller_oauthtoken type was not valid ({0}). Valid options are str or dict.'.format(type(token_param).__name__)
                self.fail_json(msg=error_msg)
        if not re.match('^https{0,1}://', self.host):
            self.host = 'https://{0}'.format(self.host)
        try:
            self.url = urlparse(self.host)
            self.url_prefix = self.url.path
        except Exception as e:
            self.fail_json(msg='Unable to parse controller_host as a URL ({1}): {0}'.format(self.host, e))
        remove_target = '[]'
        for char in remove_target:
            self.url.hostname.replace(char, '')
        try:
            proxy_env_var_name = '{0}_proxy'.format(self.url.scheme)
            if not environ.get(proxy_env_var_name) and (not environ.get(proxy_env_var_name.upper())):
                addrinfolist = getaddrinfo(self.url.hostname, self.url.port, proto=IPPROTO_TCP)
                for family, kind, proto, canonical, sockaddr in addrinfolist:
                    sockaddr[0]
        except Exception as e:
            self.fail_json(msg='Unable to resolve controller_host ({1}): {0}'.format(self.url.hostname, e))

    def build_url(self, endpoint, query_params=None):
        if not endpoint.startswith('/'):
            endpoint = '/{0}'.format(endpoint)
        prefix = self.url_prefix.rstrip('/')
        if not endpoint.startswith(prefix + '/api/'):
            endpoint = prefix + '/api/v2{0}'.format(endpoint)
        if not endpoint.endswith('/') and '?' not in endpoint:
            endpoint = '{0}/'.format(endpoint)
        url = self.url._replace(path=endpoint)
        if query_params:
            url = url._replace(query=urlencode(query_params))
        return url

    def load_config_files(self):
        config_files = ['/etc/tower/tower_cli.cfg', join(expanduser('~'), '.{0}'.format(self.config_name))]
        local_dir = getcwd()
        config_files.append(join(local_dir, self.config_name))
        while split(local_dir)[1]:
            local_dir = split(local_dir)[0]
            config_files.insert(2, join(local_dir, '.{0}'.format(self.config_name)))
        if self.params.get('controller_config_file'):
            duplicated_params = [fn for fn in self.AUTH_ARGSPEC if fn != 'controller_config_file' and self.params.get(fn) is not None]
            if duplicated_params:
                self.warn('The parameter(s) {0} were provided at the same time as controller_config_file. Precedence may be unstable, we suggest either using config file or params.'.format(', '.join(duplicated_params)))
            try:
                self.load_config(self.params.get('controller_config_file'))
            except ConfigFileException as cfe:
                self.fail_json(msg=cfe)
        else:
            for config_file in config_files:
                if exists(config_file) and (not isdir(config_file)):
                    try:
                        self.load_config(config_file)
                    except ConfigFileException:
                        self.fail_json(msg='The config file {0} is not properly formatted'.format(config_file))

    def load_config(self, config_path):
        if not isfile(config_path):
            raise ConfigFileException('The specified config file does not exist')
        if not access(config_path, R_OK):
            raise ConfigFileException('The specified config file cannot be read')
        with open(config_path, 'r') as f:
            config_string = f.read()
        try:
            try_config_parsing = True
            if HAS_YAML:
                try:
                    config_data = yaml.load(config_string, Loader=yaml.SafeLoader)
                    if type(config_data) is not dict:
                        raise AssertionError('The yaml config file is not properly formatted as a dict.')
                    try_config_parsing = False
                except (AttributeError, yaml.YAMLError, AssertionError):
                    try_config_parsing = True
            if try_config_parsing:
                if '[general]' not in config_string:
                    config_string = '[general]\n{0}'.format(config_string)
                config = ConfigParser()
                try:
                    placeholder_file = StringIO(config_string)
                    if hasattr(config, 'read_file'):
                        config.read_file(placeholder_file)
                    else:
                        config.readfp(placeholder_file)
                    config_data = {}
                    for honorred_setting in self.short_params:
                        try:
                            config_data[honorred_setting] = config.get('general', honorred_setting)
                        except NoOptionError:
                            pass
                except Exception as e:
                    raise_from(ConfigFileException('An unknown exception occured trying to ini load config file: {0}'.format(e)), e)
        except Exception as e:
            raise_from(ConfigFileException('An unknown exception occured trying to load config file: {0}'.format(e)), e)
        for honorred_setting in self.short_params:
            if honorred_setting in config_data:
                if honorred_setting == 'verify_ssl':
                    if type(config_data[honorred_setting]) is str:
                        setattr(self, honorred_setting, strtobool(config_data[honorred_setting]))
                    else:
                        setattr(self, honorred_setting, bool(config_data[honorred_setting]))
                else:
                    setattr(self, honorred_setting, config_data[honorred_setting])

    def logout(self):
        pass

    def fail_json(self, **kwargs):
        self.logout()
        if self.error_callback:
            self.error_callback(**kwargs)
        else:
            super().fail_json(**kwargs)

    def exit_json(self, **kwargs):
        self.logout()
        super().exit_json(**kwargs)

    def warn(self, warning):
        if self.warn_callback is not None:
            self.warn_callback(warning)
        else:
            super().warn(warning)