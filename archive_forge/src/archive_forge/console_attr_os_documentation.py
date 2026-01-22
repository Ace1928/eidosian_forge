from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
Reads and returns one keypress from stdin, no echo, using Windows APIs.

    Returns:
      The key name, None for EOF, <*> for function keys, otherwise a
      character.
    