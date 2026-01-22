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
def _get_client_args(self, service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token, endpoints_ruleset_data, partition_data):
    args_creator = ClientArgsCreator(self._event_emitter, self._user_agent, self._response_parser_factory, self._loader, self._exceptions_factory, config_store=self._config_store, user_agent_creator=self._user_agent_creator)
    return args_creator.get_client_args(service_model, region_name, is_secure, endpoint_url, verify, credentials, scoped_config, client_config, endpoint_bridge, auth_token, endpoints_ruleset_data, partition_data)