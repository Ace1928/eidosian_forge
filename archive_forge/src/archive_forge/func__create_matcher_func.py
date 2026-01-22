import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def _create_matcher_func(self):
    if self.matcher == 'path':
        return self._create_path_matcher()
    elif self.matcher == 'pathAll':
        return self._create_path_all_matcher()
    elif self.matcher == 'pathAny':
        return self._create_path_any_matcher()
    elif self.matcher == 'status':
        return self._create_status_matcher()
    elif self.matcher == 'error':
        return self._create_error_matcher()
    else:
        raise WaiterConfigError(error_msg='Unknown acceptor: %s' % self.matcher)