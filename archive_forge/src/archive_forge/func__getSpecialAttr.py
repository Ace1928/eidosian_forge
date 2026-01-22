import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def _getSpecialAttr(self, attr):
    return self._deferredAttr(attr)