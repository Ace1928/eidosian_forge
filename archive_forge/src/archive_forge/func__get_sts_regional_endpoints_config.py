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
def _get_sts_regional_endpoints_config(self):
    sts_regional_endpoints_config = self._config_store.get_config_variable('sts_regional_endpoints')
    if not sts_regional_endpoints_config:
        sts_regional_endpoints_config = 'legacy'
    if sts_regional_endpoints_config not in VALID_REGIONAL_ENDPOINTS_CONFIG:
        raise botocore.exceptions.InvalidSTSRegionalEndpointsConfigError(sts_regional_endpoints_config=sts_regional_endpoints_config)
    return sts_regional_endpoints_config