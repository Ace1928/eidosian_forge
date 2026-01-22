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
def _set_region_if_custom_s3_endpoint(self, endpoint_config, endpoint_bridge):
    if endpoint_config['signing_region'] is None and endpoint_config['region_name'] is None:
        endpoint = endpoint_bridge.resolve('s3')
        endpoint_config['signing_region'] = endpoint['signing_region']
        endpoint_config['region_name'] = endpoint['region_name']