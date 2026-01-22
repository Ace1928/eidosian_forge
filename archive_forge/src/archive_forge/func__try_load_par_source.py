import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def _try_load_par_source(source_file_path):
    """Try loading the source code inside a .par file.

  A .par file is a zip-compressed, self-contained Python executable.
  It contains the content of individual Python source files that can
  be read only through extracting from the zip file.

  Args:
    source_file_path: The full path to the file inside the .par file. This
      path should include the path to the .par file itself, followed by the
      intra-par path, e.g.,
      "/tmp/my_executable.par/org-tensorflow/tensorflow/python/foo/bar.py".

  Returns:
    If successful, lines of the source file as a `list` of `str`s.
    Else, `None`.
  """
    prefix_path = source_file_path
    while True:
        prefix_path, basename = os.path.split(prefix_path)
        if not basename:
            break
        suffix_path = os.path.normpath(os.path.relpath(source_file_path, start=prefix_path))
        if prefix_path.endswith('.par') and os.path.isfile(prefix_path):
            with zipfile.ZipFile(prefix_path) as z:
                norm_names = [os.path.normpath(name) for name in z.namelist()]
                if suffix_path in norm_names:
                    with z.open(z.namelist()[norm_names.index(suffix_path)]) as zf:
                        source_text = zf.read().decode('utf-8')
                        return source_text.split('\n')