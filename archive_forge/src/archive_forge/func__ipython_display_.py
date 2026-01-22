import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
def _ipython_display_(self, include=None, exclude=None):
    """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
    from IPython.display import Audio, display
    display(Audio(self.to_string(), rate=self.samplerate))