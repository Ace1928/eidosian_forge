from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
@mock.patch('keras_tuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner._prepare_model_build')
def test_tuner_not_call_super_search_with_overwrite(_, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path)
    final_fit.return_value = (mock.Mock(), mock.Mock(), mock.Mock())
    tuner.search(x=None, epochs=10, validation_data=None)
    tuner.save()
    super_search.reset_mock()
    tuner = greedy.Greedy(hypermodel=test_utils.build_graph(), directory=tmp_path)
    tuner.search(x=None, epochs=10, validation_data=None)
    super_search.assert_not_called()