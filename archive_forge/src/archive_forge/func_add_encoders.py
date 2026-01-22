import os
import math
import sys
from typing import Optional, Union, Callable
import pyglet
from pyglet.customtypes import Buffer
def add_encoders(self, module):
    """Add an encoder module.  The module must define `get_encoders`.  Once
        added, the appropriate encoders defined in the codec will be returned by
        CodecRegistry.get_encoders.
        """
    for encoder in module.get_encoders():
        self._encoders.append(encoder)
        for extension in encoder.get_file_extensions():
            if extension not in self._encoder_extensions:
                self._encoder_extensions[extension] = []
            self._encoder_extensions[extension].append(encoder)