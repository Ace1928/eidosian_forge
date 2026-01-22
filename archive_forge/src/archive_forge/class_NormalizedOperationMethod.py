import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
class NormalizedOperationMethod:

    def __init__(self, client_method):
        self._client_method = client_method

    def __call__(self, **kwargs):
        try:
            return self._client_method(**kwargs)
        except ClientError as e:
            return e.response