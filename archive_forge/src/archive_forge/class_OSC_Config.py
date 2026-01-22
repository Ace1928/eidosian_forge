import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
class OSC_Config(config.OpenStackConfig):

    def _auth_select_default_plugin(self, config):
        """Select a default plugin based on supplied arguments

        Migrated from auth.select_auth_plugin()
        """
        identity_version = str(config.get('identity_api_version', ''))
        if config.get('username', None) and (not config.get('auth_type', None)):
            if identity_version == '3':
                config['auth_type'] = 'v3password'
            elif identity_version.startswith('2'):
                config['auth_type'] = 'v2password'
            else:
                config['auth_type'] = 'password'
        elif config.get('token', None) and (not config.get('auth_type', None)):
            if identity_version == '3':
                config['auth_type'] = 'v3token'
            elif identity_version.startswith('2'):
                config['auth_type'] = 'v2token'
            else:
                config['auth_type'] = 'token'
        elif not config.get('auth_type', None):
            config['auth_type'] = 'password'
        LOG.debug('Auth plugin %s selected' % config['auth_type'])
        return config

    def _auth_v2_arguments(self, config):
        """Set up v2-required arguments from v3 info

        Migrated from auth.build_auth_params()
        """
        if 'auth_type' in config and config['auth_type'].startswith('v2'):
            if 'project_id' in config['auth']:
                config['auth']['tenant_id'] = config['auth']['project_id']
            if 'project_name' in config['auth']:
                config['auth']['tenant_name'] = config['auth']['project_name']
        return config

    def _auth_v2_ignore_v3(self, config):
        """Remove v3 arguments if present for v2 plugin

        Migrated from clientmanager.setup_auth()
        """
        if str(config.get('identity_api_version', '')).startswith('2') and config.get('auth_type').endswith('password'):
            domain_props = ['project_domain_id', 'project_domain_name', 'user_domain_id', 'user_domain_name']
            for prop in domain_props:
                if config['auth'].pop(prop, None) is not None:
                    if config.get('cloud'):
                        LOG.warning('Ignoring domain related config %s for %sbecause identity API version is 2.0' % (prop, config['cloud']))
                    else:
                        LOG.warning('Ignoring domain related config %s because identity API version is 2.0' % prop)
        return config

    def _auth_default_domain(self, config):
        """Set a default domain from available arguments

        Migrated from clientmanager.setup_auth()
        """
        identity_version = str(config.get('identity_api_version', ''))
        auth_type = config.get('auth_type', None)
        default_domain = config.get('default_domain', None)
        if identity_version == '3' and (not auth_type.startswith('v2')) and default_domain:
            if auth_type in ('password', 'v3password', 'v3totp') and (not config['auth'].get('project_domain_id')) and (not config['auth'].get('project_domain_name')):
                config['auth']['project_domain_id'] = default_domain
            if auth_type in ('password', 'v3password', 'v3totp') and (not config['auth'].get('user_domain_id')) and (not config['auth'].get('user_domain_name')):
                config['auth']['user_domain_id'] = default_domain
        return config

    def auth_config_hook(self, config):
        """Allow examination of config values before loading auth plugin

        OpenStackClient will override this to perform additional checks
        on auth_type.
        """
        config = self._auth_select_default_plugin(config)
        config = self._auth_v2_arguments(config)
        config = self._auth_v2_ignore_v3(config)
        config = self._auth_default_domain(config)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('auth_config_hook(): %s', strutils.mask_password(str(config)))
        return config

    def _validate_auth(self, config, loader, fixed_argparse=None):
        """Validate auth plugin arguments"""
        plugin_options = loader.get_options()
        msgs = []
        prompt_options = []
        for p_opt in plugin_options:
            winning_value = self._find_winning_auth_value(p_opt, config)
            if not winning_value:
                winning_value = self._find_winning_auth_value(p_opt, config['auth'])
            if not winning_value and p_opt.required:
                msgs.append('Missing value {auth_key} required for auth plugin {plugin}'.format(auth_key=p_opt.name, plugin=config.get('auth_type')))
            for opt in [p_opt.name] + [o.name for o in p_opt.deprecated]:
                opt = opt.replace('-', '_')
                config.pop(opt, None)
                config['auth'].pop(opt, None)
            if winning_value:
                if p_opt.dest is None:
                    config['auth'][p_opt.name.replace('-', '_')] = winning_value
                else:
                    config['auth'][p_opt.dest] = winning_value
            if 'prompt' in vars(p_opt) and p_opt.prompt is not None and (p_opt.dest not in config['auth']) and (self._pw_callback is not None):
                prompt_options.append(p_opt)
        if msgs:
            raise sdk_exceptions.OpenStackConfigException('\n'.join(msgs))
        else:
            for p_opt in prompt_options:
                config['auth'][p_opt.dest] = self._pw_callback(p_opt.prompt)
        return config

    def load_auth_plugin(self, config):
        """Get auth plugin and validate args"""
        loader = self._get_auth_loader(config)
        config = self._validate_auth(config, loader)
        auth_plugin = loader.load_from_options(**config['auth'])
        return auth_plugin