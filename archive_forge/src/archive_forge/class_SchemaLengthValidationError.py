from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
class SchemaLengthValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_len):
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected length: {expected_len}\n                Received value of length {len(value)}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_len):
        if len(value) != expected_len:
            raise SchemaLengthValidationError(value, full_schema, path, expected_len)