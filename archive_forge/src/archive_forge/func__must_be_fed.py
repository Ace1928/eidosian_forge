import collections
import os
import time
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import op_selector
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model.model_utils import export_output as export_output_lib
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.saved_model.model_utils.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _must_be_fed(op):
    return op.type == 'Placeholder'