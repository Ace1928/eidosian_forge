from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _randomly_zoom_inputs(self, inputs):
    inputs_shape = self.backend.shape(inputs)
    unbatched = len(inputs_shape) == 3
    if unbatched:
        inputs = self.backend.numpy.expand_dims(inputs, axis=0)
        inputs_shape = self.backend.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
        height = inputs_shape[-2]
        width = inputs_shape[-1]
    else:
        height = inputs_shape[-3]
        width = inputs_shape[-2]
    seed_generator = self._get_seed_generator(self.backend._backend)
    height_zoom = self.backend.random.uniform(minval=1.0 + self.height_lower, maxval=1.0 + self.height_upper, shape=[batch_size, 1], seed=seed_generator)
    if self.width_factor is not None:
        width_zoom = self.backend.random.uniform(minval=1.0 + self.width_lower, maxval=1.0 + self.width_upper, shape=[batch_size, 1], seed=seed_generator)
    else:
        width_zoom = height_zoom
    zooms = self.backend.cast(self.backend.numpy.concatenate([width_zoom, height_zoom], axis=1), dtype='float32')
    outputs = self.backend.image.affine_transform(inputs, transform=self._get_zoom_matrix(zooms, height, width), interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value, data_format=self.data_format)
    if unbatched:
        outputs = self.backend.numpy.squeeze(outputs, axis=0)
    return outputs