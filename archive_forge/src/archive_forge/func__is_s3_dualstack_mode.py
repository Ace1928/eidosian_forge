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
def _is_s3_dualstack_mode(self, service_name):
    if service_name not in self._DUALSTACK_CUSTOMIZED_SERVICES:
        return None
    client_config = self.client_config
    if client_config is not None and client_config.s3 is not None and ('use_dualstack_endpoint' in client_config.s3):
        return client_config.s3['use_dualstack_endpoint']
    if self.scoped_config is not None:
        enabled = self.scoped_config.get('s3', {}).get('use_dualstack_endpoint')
        if enabled in [True, 'True', 'true']:
            return True