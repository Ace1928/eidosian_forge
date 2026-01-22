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
def _is_s3_service(self, service_name):
    """Whether the service is S3 or S3 Control.

        Note that throughout this class, service_name refers to the endpoint
        prefix, not the folder name of the service in botocore/data. For
        S3 Control, the folder name is 's3control' but the endpoint prefix is
        's3-control'.
        """
    return service_name in ['s3', 's3-control']