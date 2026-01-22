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
def _ignore_configured_endpoint_urls(self, client_config):
    if client_config and client_config.ignore_configured_endpoint_urls is not None:
        return client_config.ignore_configured_endpoint_urls
    return self._config_store.get_config_variable('ignore_configured_endpoint_urls')