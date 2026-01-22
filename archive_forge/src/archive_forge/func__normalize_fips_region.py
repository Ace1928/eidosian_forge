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
def _normalize_fips_region(self, region_name, client_config):
    if region_name is not None:
        normalized_region_name = region_name.replace('fips-', '').replace('-fips', '')
        if normalized_region_name != region_name:
            config_use_fips_endpoint = Config(use_fips_endpoint=True)
            if client_config:
                client_config = client_config.merge(config_use_fips_endpoint)
            else:
                client_config = config_use_fips_endpoint
            logger.warning('transforming region from %s to %s and setting use_fips_endpoint to true. client should not be configured with a fips psuedo region.' % (region_name, normalized_region_name))
            region_name = normalized_region_name
    return (region_name, client_config)