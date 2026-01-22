import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _get_location_string(self, location):
    """Dump location of ConveterError.errors.location."""
    callstack = []
    for single_call in reversed(location.call):
        if location.type == converter_error_data_pb2.ConverterErrorData.CALLSITELOC:
            callstack.append(f'  - {single_call.source.filename}:{single_call.source.line}')
        else:
            callstack.append(str(single_call))
    callstack_dump = '\n'.join(callstack)
    return callstack_dump