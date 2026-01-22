import logging
from botocore import waiter, xform_name
from botocore.args import ClientArgsCreator
from botocore.auth import AUTH_TYPE_MAPS
from botocore.awsrequest import prepare_request_dict
from botocore.compress import maybe_compress_request
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.discovery import (
from botocore.docs.docstring import ClientMethodDocstring, PaginatorDocstring
from botocore.exceptions import (
from botocore.history import get_global_history_recorder
from botocore.hooks import first_non_none_response
from botocore.httpchecksum import (
from botocore.model import ServiceModel
from botocore.paginate import Paginator
from botocore.retries import adaptive, standard
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.exceptions import ClientError  # noqa
from botocore.utils import S3ArnParamHandler  # noqa
from botocore.utils import S3ControlArnParamHandler  # noqa
from botocore.utils import S3ControlEndpointSetter  # noqa
from botocore.utils import S3EndpointSetter  # noqa
from botocore.utils import S3RegionRedirector  # noqa
from botocore import UNSIGNED  # noqa
class ClientCreator:
    """Creates client objects for a service."""

    def __init__(self, loader, endpoint_resolver, user_agent, event_emitter, retry_handler_factory, retry_config_translator, response_parser_factory=None, exceptions_factory=None, config_store=None, user_agent_creator=None):
        self._loader = loader
        self._endpoint_resolver = endpoint_resolver
        self._user_agent = user_agent
        self._event_emitter = event_emitter
        self._retry_handler_factory = retry_handler_factory
        self._retry_config_translator = retry_config_translator
        self._response_parser_factory = response_parser_factory
        self._exceptions_factory = exceptions_factory
        self._config_store = config_store
        self._user_agent_creator = user_agent_creator

    def create_client(self, service_name, region_name, is_secure=True, endpoint_url=None, verify=None, credentials=None, scoped_config=None, api_version=None, client_config=None, auth_token=None):
        responses = self._event_emitter.emit('choose-service-name', service_name=service_name)
        service_name = first_non_none_response(responses, default=service_name)
        service_model = self._load_service_model(service_name, api_version)
        try:
            endpoints_ruleset_data = self._load_service_endpoints_ruleset(service_name, api_version)
            partition_data = self._loader.load_data('partitions')
        except UnknownServiceError:
            endpoints_ruleset_data = None
            partition_data = None
            logger.info('No endpoints ruleset found for service %s, falling back to legacy endpoint routing.', service_name)
        cls = self._create_client_class(service_name, service_model)
        region_name, client_config = self._normalize_fips_region(region_name, client_config)
        endpoint_bridge = ClientEndpointBridge(self._endpoint_resolver, scoped_config, client_config, service_signing_name=service_model.metadata.get('signingName'), config_store=self._config_store, service_signature_version=service_model.metadata.get('signatureVersion'))
        client_args = self._get_client_args(service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token, endpoints_ruleset_data, partition_data)
        service_client = cls(**client_args)
        self._register_retries(service_client)
        self._register_s3_events(client=service_client, endpoint_bridge=None, endpoint_url=None, client_config=client_config, scoped_config=scoped_config)
        self._register_s3express_events(client=service_client)
        self._register_s3_control_events(client=service_client)
        self._register_endpoint_discovery(service_client, endpoint_url, client_config)
        return service_client

    def create_client_class(self, service_name, api_version=None):
        service_model = self._load_service_model(service_name, api_version)
        return self._create_client_class(service_name, service_model)

    def _create_client_class(self, service_name, service_model):
        class_attributes = self._create_methods(service_model)
        py_name_to_operation_name = self._create_name_mapping(service_model)
        class_attributes['_PY_TO_OP_NAME'] = py_name_to_operation_name
        bases = [BaseClient]
        service_id = service_model.service_id.hyphenize()
        self._event_emitter.emit('creating-client-class.%s' % service_id, class_attributes=class_attributes, base_classes=bases)
        class_name = get_service_module_name(service_model)
        cls = type(str(class_name), tuple(bases), class_attributes)
        return cls

    def _normalize_fips_region(self, region_name, client_config):
        if region_name is not None:
            normalized_region_name = region_name.replace('fips-', '').replace('-fips', '')
            if normalized_region_name != region_name:
                config_use_fips_endpoint = Config(use_fips_endpoint=True)
                if client_config:
                    client_config = client_config.merge(config_use_fips_endpoint)
                else:
                    client_config = config_use_fips_endpoint
                logger.warning('transforming region from %s to %s and setting use_fips_endpoint to true. client should not be configured with a fips psuedo region.' % (region_name, normalized_region_name))
                region_name = normalized_region_name
        return (region_name, client_config)

    def _load_service_model(self, service_name, api_version=None):
        json_model = self._loader.load_service_model(service_name, 'service-2', api_version=api_version)
        service_model = ServiceModel(json_model, service_name=service_name)
        return service_model

    def _load_service_endpoints_ruleset(self, service_name, api_version=None):
        return self._loader.load_service_model(service_name, 'endpoint-rule-set-1', api_version=api_version)

    def _register_retries(self, client):
        retry_mode = client.meta.config.retries['mode']
        if retry_mode == 'standard':
            self._register_v2_standard_retries(client)
        elif retry_mode == 'adaptive':
            self._register_v2_standard_retries(client)
            self._register_v2_adaptive_retries(client)
        elif retry_mode == 'legacy':
            self._register_legacy_retries(client)

    def _register_v2_standard_retries(self, client):
        max_attempts = client.meta.config.retries.get('total_max_attempts')
        kwargs = {'client': client}
        if max_attempts is not None:
            kwargs['max_attempts'] = max_attempts
        standard.register_retry_handler(**kwargs)

    def _register_v2_adaptive_retries(self, client):
        adaptive.register_retry_handler(client)

    def _register_legacy_retries(self, client):
        endpoint_prefix = client.meta.service_model.endpoint_prefix
        service_id = client.meta.service_model.service_id
        service_event_name = service_id.hyphenize()
        original_config = self._loader.load_data('_retry')
        if not original_config:
            return
        retries = self._transform_legacy_retries(client.meta.config.retries)
        retry_config = self._retry_config_translator.build_retry_config(endpoint_prefix, original_config.get('retry', {}), original_config.get('definitions', {}), retries)
        logger.debug('Registering retry handlers for service: %s', client.meta.service_model.service_name)
        handler = self._retry_handler_factory.create_retry_handler(retry_config, endpoint_prefix)
        unique_id = 'retry-config-%s' % service_event_name
        client.meta.events.register(f'needs-retry.{service_event_name}', handler, unique_id=unique_id)

    def _transform_legacy_retries(self, retries):
        if retries is None:
            return
        copied_args = retries.copy()
        if 'total_max_attempts' in retries:
            copied_args = retries.copy()
            copied_args['max_attempts'] = copied_args.pop('total_max_attempts') - 1
        return copied_args

    def _get_retry_mode(self, client, config_store):
        client_retries = client.meta.config.retries
        if client_retries is not None and client_retries.get('mode') is not None:
            return client_retries['mode']
        return config_store.get_config_variable('retry_mode') or 'legacy'

    def _register_endpoint_discovery(self, client, endpoint_url, config):
        if endpoint_url is not None:
            return
        if client.meta.service_model.endpoint_discovery_operation is None:
            return
        events = client.meta.events
        service_id = client.meta.service_model.service_id.hyphenize()
        enabled = False
        if config and config.endpoint_discovery_enabled is not None:
            enabled = config.endpoint_discovery_enabled
        elif self._config_store:
            enabled = self._config_store.get_config_variable('endpoint_discovery_enabled')
        enabled = self._normalize_endpoint_discovery_config(enabled)
        if enabled and self._requires_endpoint_discovery(client, enabled):
            discover = enabled is True
            manager = EndpointDiscoveryManager(client, always_discover=discover)
            handler = EndpointDiscoveryHandler(manager)
            handler.register(events, service_id)
        else:
            events.register('before-parameter-build', block_endpoint_discovery_required_operations)

    def _normalize_endpoint_discovery_config(self, enabled):
        """Config must either be a boolean-string or string-literal 'auto'"""
        if isinstance(enabled, str):
            enabled = enabled.lower().strip()
            if enabled == 'auto':
                return enabled
            elif enabled in ('true', 'false'):
                return ensure_boolean(enabled)
        elif isinstance(enabled, bool):
            return enabled
        raise InvalidEndpointDiscoveryConfigurationError(config_value=enabled)

    def _requires_endpoint_discovery(self, client, enabled):
        if enabled == 'auto':
            return client.meta.service_model.endpoint_discovery_required
        return enabled

    def _register_eventbridge_events(self, client, endpoint_bridge, endpoint_url):
        if client.meta.service_model.service_name != 'events':
            return
        EventbridgeSignerSetter(endpoint_resolver=self._endpoint_resolver, region=client.meta.region_name, endpoint_url=endpoint_url).register(client.meta.events)

    def _register_s3express_events(self, client, endpoint_bridge=None, endpoint_url=None, client_config=None, scoped_config=None):
        if client.meta.service_model.service_name != 's3':
            return
        S3ExpressIdentityResolver(client, RefreshableCredentials).register()

    def _register_s3_events(self, client, endpoint_bridge, endpoint_url, client_config, scoped_config):
        if client.meta.service_model.service_name != 's3':
            return
        S3RegionRedirectorv2(None, client).register()
        self._set_s3_presign_signature_version(client.meta, client_config, scoped_config)

    def _register_s3_control_events(self, client, endpoint_bridge=None, endpoint_url=None, client_config=None, scoped_config=None):
        if client.meta.service_model.service_name != 's3control':
            return
        S3ControlArnParamHandlerv2().register(client.meta.events)

    def _set_s3_presign_signature_version(self, client_meta, client_config, scoped_config):
        provided_signature_version = _get_configured_signature_version('s3', client_config, scoped_config)
        if provided_signature_version is not None:
            return
        regions = self._endpoint_resolver.get_available_endpoints('s3', client_meta.partition)
        if client_meta.region_name != 'aws-global' and client_meta.region_name not in regions:
            return
        endpoint = self._endpoint_resolver.construct_endpoint('s3', client_meta.region_name)
        signature_versions = endpoint['signatureVersions']
        if 's3' not in signature_versions:
            return
        client_meta.events.register('choose-signer.s3', self._default_s3_presign_to_sigv2)

    def _default_s3_presign_to_sigv2(self, signature_version, **kwargs):
        """
        Returns the 's3' (sigv2) signer if presigning an s3 request. This is
        intended to be used to set the default signature version for the signer
        to sigv2. Situations where an asymmetric signature is required are the
        exception, for example MRAP needs v4a.

        :type signature_version: str
        :param signature_version: The current client signature version.

        :type signing_name: str
        :param signing_name: The signing name of the service.

        :return: 's3' if the request is an s3 presign request, None otherwise
        """
        if signature_version.startswith('v4a'):
            return
        if signature_version.startswith('v4-s3express'):
            return f'{signature_version}'
        for suffix in ['-query', '-presign-post']:
            if signature_version.endswith(suffix):
                return f's3{suffix}'

    def _get_client_args(self, service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token, endpoints_ruleset_data, partition_data):
        args_creator = ClientArgsCreator(self._event_emitter, self._user_agent, self._response_parser_factory, self._loader, self._exceptions_factory, config_store=self._config_store, user_agent_creator=self._user_agent_creator)
        return args_creator.get_client_args(service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token, endpoints_ruleset_data, partition_data)

    def _create_methods(self, service_model):
        op_dict = {}
        for operation_name in service_model.operation_names:
            py_operation_name = xform_name(operation_name)
            op_dict[py_operation_name] = self._create_api_method(py_operation_name, operation_name, service_model)
        return op_dict

    def _create_name_mapping(self, service_model):
        mapping = {}
        for operation_name in service_model.operation_names:
            py_operation_name = xform_name(operation_name)
            mapping[py_operation_name] = operation_name
        return mapping

    def _create_api_method(self, py_operation_name, operation_name, service_model):

        def _api_call(self, *args, **kwargs):
            if args:
                raise TypeError(f'{py_operation_name}() only accepts keyword arguments.')
            return self._make_api_call(operation_name, kwargs)
        _api_call.__name__ = str(py_operation_name)
        operation_model = service_model.operation_model(operation_name)
        docstring = ClientMethodDocstring(operation_model=operation_model, method_name=operation_name, event_emitter=self._event_emitter, method_description=operation_model.documentation, example_prefix='response = client.%s' % py_operation_name, include_signature=False)
        _api_call.__doc__ = docstring
        return _api_call