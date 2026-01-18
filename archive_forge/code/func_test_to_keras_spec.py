from tests.tune_tensorflow.mock import MockSpec
from tune_tensorflow.utils import (
def test_to_keras_spec():
    expr = to_keras_spec_expr(MockSpec)
    assert to_keras_spec(expr) == MockSpec
    expr = to_keras_spec_expr(to_keras_spec_expr(MockSpec))
    assert to_keras_spec(expr) == MockSpec