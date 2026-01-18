import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
def test_structured_data_input_unsupported_type_error():
    with pytest.raises(TypeError) as info:
        adapter = input_adapters.StructuredDataAdapter()
        adapter.adapt('unknown', batch_size=32)
    assert 'Unsupported type' in str(info.value)