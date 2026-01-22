import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
def _poll_server_till_success(max_attempts, sleep_per_poll_sec, debug_server_url, dump_dir, server, gpu_memory_fraction=1.0):
    """Poll server until success or exceeding max polling count.

  Args:
    max_attempts: (int) How many times to poll at maximum
    sleep_per_poll_sec: (float) How many seconds to sleep for after each
      unsuccessful poll.
    debug_server_url: (str) gRPC URL to the debug server.
    dump_dir: (str) Dump directory to look for files in. If None, will directly
      check data from the server object.
    server: The server object.
    gpu_memory_fraction: (float) Fraction of GPU memory to be
      allocated for the Session used in server polling.

  Returns:
    (bool) Whether the polling succeeded within max_polls attempts.
  """
    poll_count = 0
    config = config_pb2.ConfigProto(gpu_options=config_pb2.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
    with session.Session(config=config) as sess:
        for poll_count in range(max_attempts):
            server.clear_data()
            print('Polling: poll_count = %d' % poll_count)
            x_init_name = 'x_init_%d' % poll_count
            x_init = constant_op.constant([42.0], shape=[1], name=x_init_name)
            x = variables.Variable(x_init, name=x_init_name)
            run_options = config_pb2.RunOptions()
            debug_utils.add_debug_tensor_watch(run_options, x_init_name, 0, debug_urls=[debug_server_url])
            try:
                sess.run(x.initializer, options=run_options)
            except errors.FailedPreconditionError:
                pass
            if dump_dir:
                if os.path.isdir(dump_dir) and debug_data.DebugDumpDir(dump_dir).size > 0:
                    file_io.delete_recursively(dump_dir)
                    print('Poll succeeded.')
                    return True
                else:
                    print('Poll failed. Sleeping for %f s' % sleep_per_poll_sec)
                    time.sleep(sleep_per_poll_sec)
            elif server.debug_tensor_values:
                print('Poll succeeded.')
                return True
            else:
                print('Poll failed. Sleeping for %f s' % sleep_per_poll_sec)
                time.sleep(sleep_per_poll_sec)
        return False