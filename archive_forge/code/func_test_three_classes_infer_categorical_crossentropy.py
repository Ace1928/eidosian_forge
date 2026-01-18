import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import autokeras as ak
from autokeras import hyper_preprocessors
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras import test_utils
from autokeras.blocks import heads as head_module
def test_three_classes_infer_categorical_crossentropy():
    dataset = np.array(['a', 'a', 'c', 'b'])
    head = head_module.ClassificationHead(name='a', shape=(1,))
    adapter = head.get_adapter()
    dataset = adapter.adapt(dataset, batch_size=32)
    analyser = head.get_analyser()
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    head.config_from_analyser(analyser)
    head.build(keras_tuner.HyperParameters(), input_module.Input(shape=(32,)).build_node(keras_tuner.HyperParameters()))
    assert head.loss.name == 'categorical_crossentropy'