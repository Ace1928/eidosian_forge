import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils
def test_one_hot_encoder_deserialize_transforms_to_np():
    encoder = encoders.OneHotEncoder(['a', 'b', 'c'])
    encoder.fit(np.array(['a', 'b', 'a']))
    encoder = preprocessors.deserialize(preprocessors.serialize(encoder))
    one_hot = encoder.transform(tf.data.Dataset.from_tensor_slices([['a'], ['c'], ['b']]).batch(2))
    for data in one_hot:
        assert data.shape[1:] == [3]