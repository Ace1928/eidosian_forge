import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _create_saver_from_imported_meta_graph(meta_graph_def, import_scope, imported_vars):
    """Return a saver for restoring variable values to an imported MetaGraph."""
    if meta_graph_def.HasField('saver_def'):
        scope = import_scope
        var_names = list(imported_vars.keys())
        if var_names:
            sample_key = var_names[0]
            sample_var = imported_vars[sample_key]
            scope = sample_var.name[:-len(sample_key)]
        return Saver(saver_def=meta_graph_def.saver_def, name=scope)
    elif variables._all_saveable_objects(scope=import_scope):
        return Saver()
    else:
        logging.info('Saver not created because there are no variables in the graph to restore')
        return None