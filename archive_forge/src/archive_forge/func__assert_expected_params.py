import copy
from collections import deque
from pprint import pformat
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
from botocore.validate import validate_parameters
def _assert_expected_params(self, model, params, context, **kwargs):
    if self._should_not_stub(context):
        return
    self._assert_expected_call_order(model, params)
    expected_params = self._queue[0]['expected_params']
    if expected_params is None:
        return
    for param, value in expected_params.items():
        if param not in params or expected_params[param] != params[param]:
            raise StubAssertionError(operation_name=model.name, reason='Expected parameters:\n%s,\nbut received:\n%s' % (pformat(expected_params), pformat(params)))
    if sorted(expected_params.keys()) != sorted(params.keys()):
        raise StubAssertionError(operation_name=model.name, reason='Expected parameters:\n%s,\nbut received:\n%s' % (pformat(expected_params), pformat(params)))