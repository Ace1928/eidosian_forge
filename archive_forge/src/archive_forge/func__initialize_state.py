import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def _initialize_state(self, prefix=None, complex_step=False):
    if prefix is None:
        prefix = self.prefix
    if isinstance(self.initialization, Initialization):
        if not self.initialization.initialized:
            raise RuntimeError('Initialization is incomplete.')
        self._statespaces[prefix].initialize(self.initialization, complex_step=complex_step)
    else:
        raise RuntimeError('Statespace model not initialized.')