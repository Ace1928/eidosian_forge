import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def update_server_def(self, server_def, keep_alive_secs=_KEEP_ALIVE_SECS):
    """Update a server_def on the context.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.
      keep_alive_secs: Num. seconds after which the remote end will hang up. As
        long as the client is still alive, the server state for the context will
        be kept alive. If the client is killed (or there is some failure), the
        server will clean up its context keep_alive_secs after the final RPC it
        receives.

    Raises:
      ValueError: if server_def is None.
    """
    if not server_def:
        raise ValueError('server_def is None.')
    self._server_def = server_def
    if self._context_handle:
        server_def_str = server_def.SerializeToString()
        pywrap_tfe.TFE_ContextUpdateServerDef(self._context_handle, keep_alive_secs, server_def_str)
        self._initialize_logical_devices()
    self._clear_caches()