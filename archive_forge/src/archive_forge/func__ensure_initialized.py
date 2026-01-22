import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _ensure_initialized(self):
    """Initialize the async checkpoint internal state."""
    if self._initialized:
        return
    self._original_nodes = []
    self._object_map = object_identity.ObjectIdentityDictionary()
    self._tpu_embedding_objects = []
    to_traverse = collections.deque([])
    visited = object_identity.ObjectIdentitySet()
    for v in self._checkpoint_items.values():
        if isinstance(v, (Variable, ShardedVariable)):
            self._copy_trackable(v)
        elif hasattr(v, _TPU_EMBEDDING_ATTR):
            self._handle_tpu_embedding(v)
        to_traverse.append(v)
        visited.add(v)
    self._traverse_variables(to_traverse, visited)
    for current_trackable in self._original_nodes:
        if 'get_slot_names' in dir(current_trackable):
            slot_names = current_trackable.get_slot_names()
            for slot_name in slot_names:
                for original_variable in self._original_nodes:
                    if not isinstance(original_variable, Variable):
                        continue
                    try:
                        original_slot_variable = current_trackable.get_slot(original_variable, slot_name)
                    except (AttributeError, KeyError):
                        continue
                    if isinstance(original_slot_variable, (Variable, ShardedVariable)):
                        self._copy_trackable(original_slot_variable)
    save_counter = self.checkpointer().save_counter.numpy()
    logging.info("Initializing async checkpoint's save_counter: %d", save_counter)
    self.checkpointer()._saver._object_map = self._object_map
    self._async_save_thread = threading.Thread(target=self._async_save, daemon=True)
    self._async_save_thread.start()
    self._initialized = True