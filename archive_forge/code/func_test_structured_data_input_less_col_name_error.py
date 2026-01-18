import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_structured_data_input_less_col_name_error():
    with pytest.raises(ValueError) as info:
        analyser = input_analysers.StructuredDataAnalyser(column_names=list(range(8)))
        dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(20, 10)).batch(32)
        for x in dataset:
            analyser.update(x)
        analyser.finalize()
    assert 'Expect column_names to have length' in str(info.value)