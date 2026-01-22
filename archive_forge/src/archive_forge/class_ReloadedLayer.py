import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
class ReloadedLayer(base_layer.Layer):
    """Reload a Keras model/layer that was saved via SavedModel / ExportArchive.

    Arguments:
        filepath: `str` or `pathlib.Path` object. The path to the SavedModel.
        call_endpoint: Name of the endpoint to use as the `call()` method
            of the reloaded layer. If the SavedModel was created
            via `model.export()`,
            then the default endpoint name is `'serve'`. In other cases
            it may be named `'serving_default'`.

    Example:

    ```python
    model.export("path/to/artifact")
    reloaded_layer = ReloadedLayer("path/to/artifact")
    outputs = reloaded_layer(inputs)
    ```

    The reloaded object can be used like a regular Keras layer, and supports
    training/fine-tuning of its trainable weights. Note that the reloaded
    object retains none of the internal structure or custom methods of the
    original object -- it's a brand new layer created around the saved
    function.

    **Limitations:**

    * Only call endpoints with a single `inputs` tensor argument
    (which may optionally be a dict/tuple/list of tensors) are supported.
    For endpoints with multiple separate input tensor arguments, consider
    subclassing `ReloadedLayer` and implementing a `call()` method with a
    custom signature.
    * If you need training-time behavior to differ from inference-time behavior
    (i.e. if you need the reloaded object to support a `training=True` argument
    in `__call__()`), make sure that the training-time call function is
    saved as a standalone endpoint in the artifact, and provide its name
    to the `ReloadedLayer` via the `call_training_endpoint` argument.
    """

    def __init__(self, filepath, call_endpoint='serve', call_training_endpoint=None, trainable=True, name=None, dtype=None):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self._reloaded_obj = tf.saved_model.load(filepath)
        self.filepath = filepath
        self.call_endpoint = call_endpoint
        self.call_training_endpoint = call_training_endpoint
        if hasattr(self._reloaded_obj, call_endpoint):
            self.call_endpoint_fn = getattr(self._reloaded_obj, call_endpoint)
        elif call_endpoint in self._reloaded_obj.signatures:
            self.call_endpoint_fn = self._reloaded_obj.signatures[call_endpoint]
        else:
            raise ValueError(f"The endpoint '{call_endpoint}' is neither an attribute of the reloaded SavedModel, nor an entry in the `signatures` field of the reloaded SavedModel. ")
        if call_training_endpoint:
            if hasattr(self._reloaded_obj, call_training_endpoint):
                self.call_training_endpoint_fn = getattr(self._reloaded_obj, call_training_endpoint)
            elif call_training_endpoint in self._reloaded_obj.signatures:
                self.call_training_endpoint_fn = self._reloaded_obj.signatures[call_training_endpoint]
            else:
                raise ValueError(f"The endpoint '{call_training_endpoint}' is neither an attribute of the reloaded SavedModel, nor an entry in the `signatures` field of the reloaded SavedModel. ")
        all_fns = [self.call_endpoint_fn]
        if call_training_endpoint:
            all_fns.append(self.call_training_endpoint_fn)
        tvs, ntvs = _list_variables_used_by_fns(all_fns)
        for v in tvs:
            self._add_existing_weight(v, trainable=True)
        for v in ntvs:
            self._add_existing_weight(v, trainable=False)
        self.built = True

    def _add_existing_weight(self, weight, trainable):
        """Calls add_weight() to register but not create an existing weight."""
        self.add_weight(name=weight.name, shape=weight.shape, dtype=weight.dtype, trainable=trainable, getter=lambda *_, **__: weight)

    def call(self, inputs, training=False, **kwargs):
        if training:
            if self.call_training_endpoint:
                return self.call_training_endpoint_fn(inputs, **kwargs)
        return self.call_endpoint_fn(inputs, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {'filepath': self.filepath, 'call_endpoint': self.call_endpoint, 'call_training_endpoint': self.call_training_endpoint}
        return {**base_config, **config}