from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class State(object):
    """Key formats for accepting/returning state."""
    STATE_TUPLE = 'start_tuple'
    STATE_PREFIX = 'model_state'