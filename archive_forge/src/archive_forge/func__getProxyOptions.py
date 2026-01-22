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
def _getProxyOptions(self):
    return dict([(k, self._getProxyOption(k)) for k in self._proxyOptions])