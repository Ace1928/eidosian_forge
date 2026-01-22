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
def _assume_endpoint(self, service_name, region_name, endpoint_url, is_secure):
    if endpoint_url is None:
        hostname = self.default_endpoint.format(service=service_name, region=region_name)
        endpoint_url = self._make_url(hostname, is_secure, ['http', 'https'])
    logger.debug(f'Assuming an endpoint for {service_name}, {region_name}: {endpoint_url}')
    signature_version = self._resolve_signature_version(service_name, {'signatureVersions': ['v4']})
    signing_name = self._resolve_signing_name(service_name, resolved={})
    return self._create_result(service_name=service_name, region_name=region_name, signing_region=region_name, signing_name=signing_name, signature_version=signature_version, endpoint_url=endpoint_url, metadata={})