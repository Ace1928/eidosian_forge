from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class')
def test_single_nested_dataset_doesnt_crash(tuner_fn, tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, max_trials=2, overwrite=False)
    x1 = test_utils.generate_data()
    y1 = test_utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), y1))
    auto_model.fit(dataset, epochs=2)