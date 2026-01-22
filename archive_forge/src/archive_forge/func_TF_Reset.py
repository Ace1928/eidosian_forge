from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.python.client._pywrap_tf_session import _TF_SetTarget
from tensorflow.python.client._pywrap_tf_session import _TF_SetConfig
from tensorflow.python.client._pywrap_tf_session import _TF_NewSessionOptions
from tensorflow.python.util import tf_stack
def TF_Reset(target, containers=None, config=None):
    opts = TF_NewSessionOptions(target=target, config=config)
    try:
        TF_Reset_wrapper(opts, containers)
    finally:
        TF_DeleteSessionOptions(opts)