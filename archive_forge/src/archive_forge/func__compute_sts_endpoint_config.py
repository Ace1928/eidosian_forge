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
def _compute_sts_endpoint_config(self, **resolve_endpoint_kwargs):
    endpoint_config = self._resolve_endpoint(**resolve_endpoint_kwargs)
    if self._should_set_global_sts_endpoint(resolve_endpoint_kwargs['region_name'], resolve_endpoint_kwargs['endpoint_url'], endpoint_config):
        self._set_global_sts_endpoint(endpoint_config, resolve_endpoint_kwargs['is_secure'])
    return endpoint_config