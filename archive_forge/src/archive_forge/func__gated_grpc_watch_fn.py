import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
def _gated_grpc_watch_fn(fetches, feeds):
    del fetches, feeds
    return framework.WatchOptions(debug_ops=['DebugIdentity(gated_grpc=true)'])