from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import six
from six.moves import queue as Queue
from six.moves import range
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.seek_ahead_thread import SeekAheadThread
import gslib.tests.testcase as testcase
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils import unit_util
Initializes the iterator.

        Args:
          num_iterations: Total number of results to yield.
          num_iterations_before_cancel: Set cancel event before yielding
              on the given iteration.
          cancel_event: threading.Event() to signal SeekAheadThread to stop.
        