from tensorflow.python import pywrap_tfe
def watch_variable(tape, variable):
    """Marks this variable to be watched by the given tape."""
    variables = _variables_override(variable)
    for var in variables:
        pywrap_tfe.TFE_Py_TapeWatchVariable(tape._tape, var)
        pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)