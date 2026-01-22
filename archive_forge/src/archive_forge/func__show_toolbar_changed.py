import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def _show_toolbar_changed(self):
    if not self._initialized:
        return
    self.send({'type': 'change_show_toolbar'})