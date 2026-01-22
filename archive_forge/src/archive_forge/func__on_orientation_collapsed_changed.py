import sys
import types
from warnings import warn
import io
import json
from base64 import b64encode
import matplotlib
import numpy as np
from IPython import get_ipython
from IPython import version_info as ipython_version_info
from IPython.display import HTML, display
from ipython_genutils.py3compat import string_types
from ipywidgets import DOMWidget, widget_serialization
from matplotlib import is_interactive, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import NavigationToolbar2, _Backend, cursors
from matplotlib.backends.backend_webagg_core import (
from PIL import Image
from traitlets import (
from ._version import js_semver
@observe('orientation', 'collapsed')
def _on_orientation_collapsed_changed(self, change):
    warn("The Toolbar properties 'orientation' and 'collapsed' are deprecated.", DeprecationWarning)