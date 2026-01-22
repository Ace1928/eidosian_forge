from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import io
Initializes the FilePart.

    Args:
      filename: The name of the existing file, of which this object represents
                a part.
      offset: The position (in bytes) in the original file that corresponds to
              the first byte of the FilePart.
      length: The total number of bytes in the FilePart.
    