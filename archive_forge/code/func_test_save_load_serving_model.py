import random
import tempfile
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import string_lookup
def test_save_load_serving_model(self, model, feature_mapper, label_inverse_lookup_layer):
    """Test save/load/serving model."""
    serving_fn = self.create_serving_signature(model, feature_mapper, label_inverse_lookup_layer)
    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(saved_model_dir, save_format='tf', signatures={'serving_default': serving_fn})
    loaded_serving_fn = keras.saving.legacy.save.load_model(saved_model_dir).signatures['serving_default']
    prediction0 = loaded_serving_fn(tf.constant(['avenger', 'ironman', 'avenger']))['output_0']
    self.assertIn(prediction0.numpy().decode('UTF-8'), ('yes', 'no'))
    prediction1 = loaded_serving_fn(tf.constant(['ironman', 'ironman', 'unknown']))['output_0']
    self.assertIn(prediction1.numpy().decode('UTF-8'), ('yes', 'no'))