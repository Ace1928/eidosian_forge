import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_specify_two_classes_fit_single_column():
    analyser = output_analysers.ClassificationAnalyser(name='a', num_classes=2)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.num_classes == 2