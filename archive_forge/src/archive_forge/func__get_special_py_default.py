from botocore.utils import is_json_value_header
def _get_special_py_default(self, shape):
    special_defaults = {'document_type': "{...}|[...]|123|123.4|'string'|True|None", 'jsonvalue_header': "{...}|[...]|123|123.4|'string'|True|None", 'streaming_input_shape': "b'bytes'|file", 'streaming_output_shape': 'StreamingBody()', 'eventstream_output_shape': 'EventStream()'}
    return self._get_value_for_special_type(shape, special_defaults)