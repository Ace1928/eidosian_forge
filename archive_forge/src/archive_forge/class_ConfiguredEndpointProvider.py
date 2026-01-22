import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class ConfiguredEndpointProvider(BaseProvider):
    """Lookup an endpoint URL from environment variable or shared config file.

    NOTE: This class is considered private and is subject to abrupt breaking
    changes or removal without prior announcement. Please do not use it
    directly.
    """
    _ENDPOINT_URL_LOOKUP_ORDER = ['environment_service', 'environment_global', 'config_service', 'config_global']

    def __init__(self, full_config, scoped_config, client_name, environ=None):
        """Initialize a ConfiguredEndpointProviderChain.

        :type full_config: dict
        :param full_config: This is the dict representing the full
            configuration file.

        :type scoped_config: dict
        :param scoped_config: This is the dict representing the configuration
            for the current profile for the session.

        :type client_name: str
        :param client_name: The name used to instantiate a client using
            botocore.session.Session.create_client.

        :type environ: dict
        :param environ: A mapping to use for environment variables. If this
            is not provided it will default to use os.environ.
        """
        self._full_config = full_config
        self._scoped_config = scoped_config
        self._client_name = client_name
        self._transformed_service_id = self._get_snake_case_service_id(self._client_name)
        if environ is None:
            environ = os.environ
        self._environ = environ

    def provide(self):
        """Lookup the configured endpoint URL.

        The order is:

        1. The value provided by a service-specific environment variable.
        2. The value provided by the global endpoint environment variable
           (AWS_ENDPOINT_URL).
        3. The value provided by a service-specific parameter from a services
           definition section in the shared configuration file.
        4. The value provided by the global parameter from a services
           definition section in the shared configuration file.
        """
        for location in self._ENDPOINT_URL_LOOKUP_ORDER:
            logger.debug('Looking for endpoint for %s via: %s', self._client_name, location)
            endpoint_url = getattr(self, f'_get_endpoint_url_{location}')()
            if endpoint_url:
                logger.info('Found endpoint for %s via: %s.', self._client_name, location)
                return endpoint_url
        logger.debug('No configured endpoint found.')
        return None

    def _get_snake_case_service_id(self, client_name):
        client_name = utils.SERVICE_NAME_ALIASES.get(client_name, client_name)
        hyphenized_service_id = utils.CLIENT_NAME_TO_HYPHENIZED_SERVICE_ID_OVERRIDES.get(client_name, client_name)
        return hyphenized_service_id.replace('-', '_')

    def _get_service_env_var_name(self):
        transformed_service_id_env = self._transformed_service_id.upper()
        return f'AWS_ENDPOINT_URL_{transformed_service_id_env}'

    def _get_services_config(self):
        if 'services' not in self._scoped_config:
            return {}
        section_name = self._scoped_config['services']
        services_section = self._full_config.get('services', {}).get(section_name)
        if not services_section:
            error_msg = f'The profile is configured to use the services section but the "{section_name}" services configuration does not exist.'
            raise InvalidConfigError(error_msg=error_msg)
        return services_section

    def _get_endpoint_url_config_service(self):
        snakecase_service_id = self._transformed_service_id.lower()
        return self._get_services_config().get(snakecase_service_id, {}).get('endpoint_url')

    def _get_endpoint_url_config_global(self):
        return self._scoped_config.get('endpoint_url')

    def _get_endpoint_url_environment_service(self):
        return EnvironmentProvider(name=self._get_service_env_var_name(), env=self._environ).provide()

    def _get_endpoint_url_environment_global(self):
        return EnvironmentProvider(name='AWS_ENDPOINT_URL', env=self._environ).provide()