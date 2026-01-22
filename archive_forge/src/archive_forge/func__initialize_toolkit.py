import sys
import os
from os import path
from contextlib import contextmanager
def _initialize_toolkit(self):
    """
        Initializes the toolkit.

        """
    if self._toolkit is not None:
        toolkit = self._toolkit
    else:
        toolkit = os.environ.get('ETS_TOOLKIT', '')
    return toolkit