import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_unrecognized_arg():
    """Getting to some parts of the codes imply that the
    set of supported of languages has changed.  Modify the
    supported languages to simulate this future code change."""
    cirq_google.serialization.arg_func_langs.SUPPORTED_FUNCTIONS_FOR_LANGUAGE['test'] = frozenset({'magic'})
    with pytest.raises(ValueError, match='could not be processed'):
        _ = float_arg_from_proto(v2.program_pb2.Arg(func=v2.program_pb2.ArgFunction(type='magic')), arg_function_language='test', required_arg_name='blah')
    del cirq_google.serialization.arg_func_langs.SUPPORTED_FUNCTIONS_FOR_LANGUAGE['test']