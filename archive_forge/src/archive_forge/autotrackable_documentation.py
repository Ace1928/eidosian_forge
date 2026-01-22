import warnings
from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.types import core as core_types
from tensorflow.python.util.tf_export import tf_export
Removes the tracking of name.