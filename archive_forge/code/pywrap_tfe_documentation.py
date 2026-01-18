from tensorflow.python import pywrap_tensorflow
from tensorflow.python._pywrap_tfe import *
Python module for TFE ops and functions exported by pybind11.

This module is created because we are splitting out eager bindings from
pywrap_tensorflow. This is causing some issues where Graphs are not properly
initialized when running eager code. Once the graph architecture has been
removed from pywrap_tensorflow as well, we can remove this file.
