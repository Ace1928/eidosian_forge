import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_text_input_with_illegal_dim():
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the data to TextInput to have shape' in str(info.value)