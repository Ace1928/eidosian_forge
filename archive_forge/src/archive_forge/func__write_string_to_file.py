import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _write_string_to_file(filename, file_content):
    """Writes a string to a given file.

    Args:
      filename: string, path to a file
      file_content: string, contents that need to be written to the file

    Raises:
      errors.OpError: If there are errors during the operation.
    """
    with GFile(filename, mode='w') as f:
        f.write(compat.as_text(file_content))