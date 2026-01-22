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
def _convert_to_request_dict(self, api_params, operation_model, endpoint_url, context=None, headers=None, set_user_agent_header=True):
    request_dict = self._serializer.serialize_to_request(api_params, operation_model)
    if not self._client_config.inject_host_prefix:
        request_dict.pop('host_prefix', None)
    if headers is not None:
        request_dict['headers'].update(headers)
    if set_user_agent_header:
        user_agent = self._user_agent_creator.to_string()
    else:
        user_agent = None
    prepare_request_dict(request_dict, endpoint_url=endpoint_url, user_agent=user_agent, context=context)
    return request_dict