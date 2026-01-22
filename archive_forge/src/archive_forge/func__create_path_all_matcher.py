import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def _create_path_all_matcher(self):
    expression = jmespath.compile(self.argument)
    expected = self.expected

    def acceptor_matches(response):
        if is_valid_waiter_error(response):
            return
        result = expression.search(response)
        if not isinstance(result, list) or not result:
            return False
        for element in result:
            if element != expected:
                return False
        return True
    return acceptor_matches