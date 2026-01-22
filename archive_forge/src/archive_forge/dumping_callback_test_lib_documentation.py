import os
import shutil
import tempfile
import uuid
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
Read and check the .metadata debug-events file.