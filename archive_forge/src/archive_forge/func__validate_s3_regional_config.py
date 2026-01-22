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
def _validate_s3_regional_config(self, config_val):
    if config_val not in VALID_REGIONAL_ENDPOINTS_CONFIG:
        raise botocore.exceptions.InvalidS3UsEast1RegionalEndpointConfigError(s3_us_east_1_regional_endpoint_config=config_val)