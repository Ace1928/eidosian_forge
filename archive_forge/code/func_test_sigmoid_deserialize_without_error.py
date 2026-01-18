import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import postprocessors
def test_sigmoid_deserialize_without_error():
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = tf.data.Dataset.from_tensor_slices([1, 2]).batch(32)
    postprocessor = preprocessors.deserialize(preprocessors.serialize(postprocessor))
    assert postprocessor.transform(dataset) is dataset