import base64
import binascii
import re
import string
import six
def Chunk(value, size, start=0):
    """Break a string into chunks of a given size.

  Args:
    value: The value to split.
    size: The maximum size of a chunk.
    start: The index at which to start (defaults to 0).

  Returns:
    Iterable over string slices of as close to the given size as possible.
    Chunk('hello', 2) => 'he', 'll', 'o'

  Raises:
    ValueError: If start < 0 or if size <= 0.
  """
    if start < 0:
        raise ValueError('invalid starting position')
    if size <= 0:
        raise ValueError('invalid chunk size')
    return (value[i:i + size] for i in range(start, len(value), size))