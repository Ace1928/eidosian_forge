from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class', side_effect=get_tuner_class)
def test_predict_tuple_x_and_tuple_y_predict_doesnt_crash(tuner_fn, tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path)
    dataset = tf.data.Dataset.from_tensor_slices(((np.random.rand(100, 32, 32, 3),), (np.random.rand(100, 1),)))
    auto_model.fit(dataset)
    auto_model.predict(dataset)