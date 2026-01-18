from tensorflow.python import pywrap_tfe
def variables_accessed(variables):
    """Notifies all tapes in the stack that variables have been accessed.

  Only trainable variables are marked as accessed.

  Args:
    variables: iterable of variables to mark as accessed.
  """
    accessed = []
    for variable in variables:
        if variable.trainable:
            accessed.extend(_variables_override(variable))
    for var in accessed:
        pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
        pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)