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
def _get_configured_signature_version(service_name, client_config, scoped_config):
    """
    Gets the manually configured signature version.

    :returns: the customer configured signature version, or None if no
        signature version was configured.
    """
    if client_config and client_config.signature_version is not None:
        return client_config.signature_version
    if scoped_config is not None:
        service_config = scoped_config.get(service_name)
        if service_config is not None and isinstance(service_config, dict):
            version = service_config.get('signature_version')
            if version:
                logger.debug('Switching signature version for service %s to version %s based on config file override.', service_name, version)
                return version
    return None