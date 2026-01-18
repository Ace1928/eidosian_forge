import contextlib
from tensorflow.python import pywrap_tfe
def watched_variables(self):
    """Returns a tuple of variables accessed under this scope."""
    return pywrap_tfe.TFE_Py_VariableWatcherWatchedVariables(self._variable_watcher)