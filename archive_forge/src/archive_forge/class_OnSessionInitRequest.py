import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class OnSessionInitRequest:
    """Request to an on-session-init callback.

  This callback is invoked during the __init__ call to a debug-wrapper session.
  """

    def __init__(self, sess):
        """Constructor.

    Args:
      sess: A tensorflow Session object.
    """
        _check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))
        self.session = sess