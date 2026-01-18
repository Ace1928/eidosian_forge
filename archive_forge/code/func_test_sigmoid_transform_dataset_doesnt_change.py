import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import postprocessors
def test_sigmoid_transform_dataset_doesnt_change():
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    assert postprocessor.transform(dataset) is dataset