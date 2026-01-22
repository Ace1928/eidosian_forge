import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class SmartDefaultsConfigStoreFactory:

    def __init__(self, default_config_resolver, imds_region_provider):
        self._default_config_resolver = default_config_resolver
        self._imds_region_provider = imds_region_provider
        self._instance_metadata_region = None

    def merge_smart_defaults(self, config_store, mode, region_name):
        if mode == 'auto':
            mode = self.resolve_auto_mode(region_name)
        default_configs = self._default_config_resolver.get_default_config_values(mode)
        for config_var in default_configs:
            config_value = default_configs[config_var]
            method = getattr(self, f'_set_{config_var}', None)
            if method:
                method(config_store, config_value)

    def resolve_auto_mode(self, region_name):
        current_region = None
        if os.environ.get('AWS_EXECUTION_ENV'):
            default_region = os.environ.get('AWS_DEFAULT_REGION')
            current_region = os.environ.get('AWS_REGION', default_region)
        if not current_region:
            if self._instance_metadata_region:
                current_region = self._instance_metadata_region
            else:
                try:
                    current_region = self._imds_region_provider.provide()
                    self._instance_metadata_region = current_region
                except Exception:
                    pass
        if current_region:
            if region_name == current_region:
                return 'in-region'
            else:
                return 'cross-region'
        return 'standard'

    def _update_provider(self, config_store, variable, value):
        original_provider = config_store.get_config_provider(variable)
        default_provider = ConstantProvider(value)
        if isinstance(original_provider, ChainProvider):
            chain_provider_copy = copy.deepcopy(original_provider)
            chain_provider_copy.set_default_provider(default_provider)
            default_provider = chain_provider_copy
        elif isinstance(original_provider, BaseProvider):
            default_provider = ChainProvider(providers=[original_provider, default_provider])
        config_store.set_config_provider(variable, default_provider)

    def _update_section_provider(self, config_store, section_name, variable, value):
        section_provider_copy = copy.deepcopy(config_store.get_config_provider(section_name))
        section_provider_copy.set_default_provider(variable, ConstantProvider(value))
        config_store.set_config_provider(section_name, section_provider_copy)

    def _set_retryMode(self, config_store, value):
        self._update_provider(config_store, 'retry_mode', value)

    def _set_stsRegionalEndpoints(self, config_store, value):
        self._update_provider(config_store, 'sts_regional_endpoints', value)

    def _set_s3UsEast1RegionalEndpoints(self, config_store, value):
        self._update_section_provider(config_store, 's3', 'us_east_1_regional_endpoint', value)

    def _set_connectTimeoutInMillis(self, config_store, value):
        self._update_provider(config_store, 'connect_timeout', value / 1000)