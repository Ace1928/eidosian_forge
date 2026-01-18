from tensorflow.python import pywrap_tfe
def watch(tape, tensor):
    """Marks this tensor to be watched by the given tape."""
    pywrap_tfe.TFE_Py_TapeWatch(tape._tape, tensor)