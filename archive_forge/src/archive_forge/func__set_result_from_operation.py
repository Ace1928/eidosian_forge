import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def _set_result_from_operation(self):
    """Set the result or exception from the operation if it is complete."""
    with self._completion_lock:
        if not self._operation.done or self._result_set:
            return
        if self._operation.HasField('response'):
            response = protobuf_helpers.from_any_pb(self._result_type, self._operation.response)
            self.set_result(response)
        elif self._operation.HasField('error'):
            exception = exceptions.from_grpc_status(status_code=self._operation.error.code, message=self._operation.error.message, errors=(self._operation.error,), response=self._operation)
            self.set_exception(exception)
        else:
            exception = exceptions.GoogleAPICallError('Unexpected state: Long-running operation had neither response nor error set.')
            self.set_exception(exception)