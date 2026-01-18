import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def write_out(self, filepath, options=None):
    """Write the corresponding SavedModel to disk.

        Arguments:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the artifact.
            options: `tf.saved_model.SaveOptions` object that specifies
                SavedModel saving options.

        **Note on TF-Serving**: all endpoints registered via `add_endpoint()`
        are made visible for TF-Serving in the SavedModel artifact. In addition,
        the first endpoint registered is made visible under the alias
        `"serving_default"` (unless an endpoint with the name
        `"serving_default"` was already registered manually),
        since TF-Serving requires this endpoint to be set.
        """
    if not self._endpoint_names:
        raise ValueError('No endpoints have been set yet. Call add_endpoint().')
    self._filter_and_track_resources()
    signatures = {}
    for name in self._endpoint_names:
        signatures[name] = self._get_concrete_fn(name)
    if 'serving_default' not in self._endpoint_names:
        signatures['serving_default'] = self._get_concrete_fn(self._endpoint_names[0])
    tf.saved_model.save(self, filepath, options=options, signatures=signatures)
    endpoints = '\n\n'.join((_print_signature(getattr(self, name), name) for name in self._endpoint_names))
    io_utils.print_msg(f"Saved artifact at '{filepath}'. The following endpoints are available:\n\n{endpoints}")