import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _read_file_to_string(filename, binary_mode=False):
    """Reads the entire contents of a file to a string.

    Args:
      filename: string, path to a file
      binary_mode: whether to open the file in binary mode or not. This changes
        the type of the object returned.

    Returns:
      contents of the file as a string or bytes.

    Raises:
      errors.OpError: Raises variety of errors that are subtypes e.g.
      `NotFoundError` etc.
    """
    if binary_mode:
        f = GFile(filename, mode='rb')
    else:
        f = GFile(filename, mode='r')
    return f.read()