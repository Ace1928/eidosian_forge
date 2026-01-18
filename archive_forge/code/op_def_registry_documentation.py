import threading
from tensorflow.core.framework import op_def_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_registry
No-op. Used to synchronize the contents of the Python registry with C++.