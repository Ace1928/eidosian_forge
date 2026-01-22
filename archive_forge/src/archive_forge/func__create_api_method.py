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
def _create_api_method(self, py_operation_name, operation_name, service_model):

    def _api_call(self, *args, **kwargs):
        if args:
            raise TypeError(f'{py_operation_name}() only accepts keyword arguments.')
        return self._make_api_call(operation_name, kwargs)
    _api_call.__name__ = str(py_operation_name)
    operation_model = service_model.operation_model(operation_name)
    docstring = ClientMethodDocstring(operation_model=operation_model, method_name=operation_name, event_emitter=self._event_emitter, method_description=operation_model.documentation, example_prefix='response = client.%s' % py_operation_name, include_signature=False)
    _api_call.__doc__ = docstring
    return _api_call