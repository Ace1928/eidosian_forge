import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def _compute_s3_endpoint_config(self, s3_config, **resolve_endpoint_kwargs):
    force_s3_global = self._should_force_s3_global(resolve_endpoint_kwargs['region_name'], s3_config)
    if force_s3_global:
        resolve_endpoint_kwargs['region_name'] = None
    endpoint_config = self._resolve_endpoint(**resolve_endpoint_kwargs)
    self._set_region_if_custom_s3_endpoint(endpoint_config, resolve_endpoint_kwargs['endpoint_bridge'])
    if force_s3_global and endpoint_config['region_name'] == 'aws-global':
        endpoint_config['region_name'] = 'us-east-1'
    return endpoint_config