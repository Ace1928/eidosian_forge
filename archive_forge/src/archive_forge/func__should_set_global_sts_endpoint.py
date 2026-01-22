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
def _should_set_global_sts_endpoint(self, region_name, endpoint_url, endpoint_config):
    has_variant_tags = endpoint_config and endpoint_config.get('metadata', {}).get('tags')
    if endpoint_url or has_variant_tags:
        return False
    return self._get_sts_regional_endpoints_config() == 'legacy' and region_name in LEGACY_GLOBAL_STS_REGIONS