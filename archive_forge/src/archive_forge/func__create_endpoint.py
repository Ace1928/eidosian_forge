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
def _create_endpoint(self, resolved, service_name, region_name, endpoint_url, is_secure):
    region_name, signing_region = self._pick_region_values(resolved, region_name, endpoint_url)
    if endpoint_url is None:
        endpoint_url = self._make_url(resolved.get('hostname'), is_secure, resolved.get('protocols', []))
    signature_version = self._resolve_signature_version(service_name, resolved)
    signing_name = self._resolve_signing_name(service_name, resolved)
    return self._create_result(service_name=service_name, region_name=region_name, signing_region=signing_region, signing_name=signing_name, endpoint_url=endpoint_url, metadata=resolved, signature_version=signature_version)