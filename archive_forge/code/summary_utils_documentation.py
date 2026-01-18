import math
import re
import shutil
import rich
import rich.console
import rich.markup
import rich.table
import tree
from keras.src import backend
from keras.src.utils import dtype_utils
from keras.src.utils import io_utils
Get the layer indexes from the model based on layer names.

    The layer indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `Model` instance.
        layer_names: a list or tuple of 2 strings, the starting layer name and
            ending layer name (both inclusive) for the result. All layers will
            be included when `None` is provided.

    Returns:
        The index value of layer based on its unique name (layer_names).
        Output will be [first_layer_index, last_layer_index + 1].
    