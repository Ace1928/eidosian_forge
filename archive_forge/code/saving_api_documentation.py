import os
import warnings
import zipfile
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.saving import saving_lib
from keras.src.saving.legacy import save as legacy_sm_saving_lib
from keras.src.utils import io_utils
Loads a model saved via `model.save()`.

    Args:
        filepath: `str` or `pathlib.Path` object, path to the saved model file.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model after loading.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.
            When `safe_mode=False`, loading an object has the potential to
            trigger arbitrary code execution. This argument is only
            applicable to the Keras v3 model format. Defaults to True.

    SavedModel format arguments:
        options: Only applies to SavedModel format.
            Optional `tf.saved_model.LoadOptions` object that specifies
            SavedModel loading options.

    Returns:
        A Keras model instance. If the original model was compiled,
        and the argument `compile=True` is set, then the returned model
        will be compiled. Otherwise, the model will be left uncompiled.

    Example:

    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_shape=(3,)),
        tf.keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = tf.keras.saving.load_model("model.keras")
    x = tf.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that the model variables may have different name values
    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
    It is recommended that you use layer attributes to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.
    