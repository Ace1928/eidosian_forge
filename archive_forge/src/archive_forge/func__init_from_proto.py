from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import threading
import tensorflow as tf
def _init_from_proto(self, queue_runner_def):
    raise NotImplementedError('{} does not support initialization from proto.'.format(type(self).__name__))