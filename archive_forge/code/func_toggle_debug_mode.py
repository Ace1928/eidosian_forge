from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export
def toggle_debug_mode(debug_mode):
    global DEBUG_MODE
    DEBUG_MODE = debug_mode