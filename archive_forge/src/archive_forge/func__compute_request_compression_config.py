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
def _compute_request_compression_config(self, config_kwargs):
    min_size = config_kwargs.get('request_min_compression_size_bytes')
    disabled = config_kwargs.get('disable_request_compression')
    if min_size is None:
        min_size = self._config_store.get_config_variable('request_min_compression_size_bytes')
    min_size = self._validate_min_compression_size(min_size)
    config_kwargs['request_min_compression_size_bytes'] = min_size
    if disabled is None:
        disabled = self._config_store.get_config_variable('disable_request_compression')
    else:
        disabled = ensure_boolean(disabled)
    config_kwargs['disable_request_compression'] = disabled