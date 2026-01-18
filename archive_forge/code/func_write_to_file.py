import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def write_to_file(self, file_path):
    """Write the object itself to file, in a plain format.

    The font_attr_segs and annotations are ignored.

    Args:
      file_path: (str) path of the file to write to.
    """
    with gfile.Open(file_path, 'w') as f:
        for line in self._lines:
            f.write(line + '\n')