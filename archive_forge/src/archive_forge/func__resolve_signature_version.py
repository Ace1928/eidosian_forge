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
def _resolve_signature_version(self, service_name, resolved):
    configured_version = _get_configured_signature_version(service_name, self.client_config, self.scoped_config)
    if configured_version is not None:
        return configured_version
    potential_versions = resolved.get('signatureVersions', [])
    if self.service_signature_version is not None and self.service_signature_version not in _LEGACY_SIGNATURE_VERSIONS:
        potential_versions = [self.service_signature_version]
    if 'signatureVersions' in resolved:
        if service_name == 's3':
            return 's3v4'
        if 'v4' in potential_versions:
            return 'v4'
        for known in potential_versions:
            if known in AUTH_TYPE_MAPS:
                return known
    raise UnknownSignatureVersionError(signature_version=potential_versions)