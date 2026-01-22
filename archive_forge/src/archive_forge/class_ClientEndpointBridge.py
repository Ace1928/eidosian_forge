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
class ClientEndpointBridge:
    """Bridges endpoint data and client creation

    This class handles taking out the relevant arguments from the endpoint
    resolver and determining which values to use, taking into account any
    client configuration options and scope configuration options.

    This class also handles determining what, if any, region to use if no
    explicit region setting is provided. For example, Amazon S3 client will
    utilize "us-east-1" by default if no region can be resolved."""
    DEFAULT_ENDPOINT = '{service}.{region}.amazonaws.com'
    _DUALSTACK_CUSTOMIZED_SERVICES = ['s3', 's3-control']

    def __init__(self, endpoint_resolver, scoped_config=None, client_config=None, default_endpoint=None, service_signing_name=None, config_store=None, service_signature_version=None):
        self.service_signing_name = service_signing_name
        self.endpoint_resolver = endpoint_resolver
        self.scoped_config = scoped_config
        self.client_config = client_config
        self.default_endpoint = default_endpoint or self.DEFAULT_ENDPOINT
        self.config_store = config_store
        self.service_signature_version = service_signature_version

    def resolve(self, service_name, region_name=None, endpoint_url=None, is_secure=True):
        region_name = self._check_default_region(service_name, region_name)
        use_dualstack_endpoint = self._resolve_use_dualstack_endpoint(service_name)
        use_fips_endpoint = self._resolve_endpoint_variant_config_var('use_fips_endpoint')
        resolved = self.endpoint_resolver.construct_endpoint(service_name, region_name, use_dualstack_endpoint=use_dualstack_endpoint, use_fips_endpoint=use_fips_endpoint)
        if not resolved:
            resolved = self.endpoint_resolver.construct_endpoint(service_name, region_name, partition_name='aws', use_dualstack_endpoint=use_dualstack_endpoint, use_fips_endpoint=use_fips_endpoint)
        if resolved:
            return self._create_endpoint(resolved, service_name, region_name, endpoint_url, is_secure)
        else:
            return self._assume_endpoint(service_name, region_name, endpoint_url, is_secure)

    def resolver_uses_builtin_data(self):
        return self.endpoint_resolver.uses_builtin_data

    def _check_default_region(self, service_name, region_name):
        if region_name is not None:
            return region_name
        if self.client_config and self.client_config.region_name is not None:
            return self.client_config.region_name

    def _create_endpoint(self, resolved, service_name, region_name, endpoint_url, is_secure):
        region_name, signing_region = self._pick_region_values(resolved, region_name, endpoint_url)
        if endpoint_url is None:
            endpoint_url = self._make_url(resolved.get('hostname'), is_secure, resolved.get('protocols', []))
        signature_version = self._resolve_signature_version(service_name, resolved)
        signing_name = self._resolve_signing_name(service_name, resolved)
        return self._create_result(service_name=service_name, region_name=region_name, signing_region=signing_region, signing_name=signing_name, endpoint_url=endpoint_url, metadata=resolved, signature_version=signature_version)

    def _resolve_endpoint_variant_config_var(self, config_var):
        client_config = self.client_config
        config_val = False
        if client_config and getattr(client_config, config_var) is not None:
            return getattr(client_config, config_var)
        elif self.config_store is not None:
            config_val = self.config_store.get_config_variable(config_var)
        return config_val

    def _resolve_use_dualstack_endpoint(self, service_name):
        s3_dualstack_mode = self._is_s3_dualstack_mode(service_name)
        if s3_dualstack_mode is not None:
            return s3_dualstack_mode
        return self._resolve_endpoint_variant_config_var('use_dualstack_endpoint')

    def _is_s3_dualstack_mode(self, service_name):
        if service_name not in self._DUALSTACK_CUSTOMIZED_SERVICES:
            return None
        client_config = self.client_config
        if client_config is not None and client_config.s3 is not None and ('use_dualstack_endpoint' in client_config.s3):
            return client_config.s3['use_dualstack_endpoint']
        if self.scoped_config is not None:
            enabled = self.scoped_config.get('s3', {}).get('use_dualstack_endpoint')
            if enabled in [True, 'True', 'true']:
                return True

    def _assume_endpoint(self, service_name, region_name, endpoint_url, is_secure):
        if endpoint_url is None:
            hostname = self.default_endpoint.format(service=service_name, region=region_name)
            endpoint_url = self._make_url(hostname, is_secure, ['http', 'https'])
        logger.debug(f'Assuming an endpoint for {service_name}, {region_name}: {endpoint_url}')
        signature_version = self._resolve_signature_version(service_name, {'signatureVersions': ['v4']})
        signing_name = self._resolve_signing_name(service_name, resolved={})
        return self._create_result(service_name=service_name, region_name=region_name, signing_region=region_name, signing_name=signing_name, signature_version=signature_version, endpoint_url=endpoint_url, metadata={})

    def _create_result(self, service_name, region_name, signing_region, signing_name, endpoint_url, signature_version, metadata):
        return {'service_name': service_name, 'region_name': region_name, 'signing_region': signing_region, 'signing_name': signing_name, 'endpoint_url': endpoint_url, 'signature_version': signature_version, 'metadata': metadata}

    def _make_url(self, hostname, is_secure, supported_protocols):
        if is_secure and 'https' in supported_protocols:
            scheme = 'https'
        else:
            scheme = 'http'
        return f'{scheme}://{hostname}'

    def _resolve_signing_name(self, service_name, resolved):
        if 'credentialScope' in resolved and 'service' in resolved['credentialScope']:
            return resolved['credentialScope']['service']
        if self.service_signing_name:
            return self.service_signing_name
        return service_name

    def _pick_region_values(self, resolved, region_name, endpoint_url):
        signing_region = region_name
        if endpoint_url is None:
            region_name = resolved['endpointName']
            signing_region = region_name
            if 'credentialScope' in resolved and 'region' in resolved['credentialScope']:
                signing_region = resolved['credentialScope']['region']
        return (region_name, signing_region)

    def _resolve_signature_version(self, service_name, resolved):
        configured_version = _get_configured_signature_version(service_name, self.client_config, self.scoped_config)
        if configured_version is not None:
            return configured_version
        potential_versions = resolved.get('signatureVersions', [])
        if self.service_signature_version is not None and self.service_signature_version not in _LEGACY_SIGNATURE_VERSIONS:
            potential_versions = [self.service_signature_version]
        if 'signatureVersions' in resolved:
            if service_name == 's3':
                return 's3v4'
            if 'v4' in potential_versions:
                return 'v4'
            for known in potential_versions:
                if known in AUTH_TYPE_MAPS:
                    return known
        raise UnknownSignatureVersionError(signature_version=potential_versions)