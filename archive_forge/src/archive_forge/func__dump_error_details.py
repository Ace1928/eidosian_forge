import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _dump_error_details(self, ops, locations):
    """Dump the list of ops and locations."""
    for i in range(0, len(ops)):
        callstack_dump = self._get_location_string(locations[i])
        err_string = f'Op: {ops[i]}\n{callstack_dump}\n'
        self._log(err_string)