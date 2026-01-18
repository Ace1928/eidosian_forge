import os
import sys
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
Converts a Keras model to dot format and save to a file.

    Example:

    ```python
    inputs = ...
    outputs = ...
    model = keras.Model(inputs=inputs, outputs=outputs)

    dot_img_file = '/tmp/model_1.png'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    ```

    Args:
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_dtype: whether to display layer dtypes.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot: `"TB"`
            creates a vertical plot; `"LR"` creates a horizontal plot.
        expand_nested: whether to expand nested Functional models
            into clusters.
        dpi: Image resolution in dots per inch.
        show_layer_activations: Display layer activations (only for layers that
            have an `activation` property).
        show_trainable: whether to display if a layer is trainable.

    Returns:
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    