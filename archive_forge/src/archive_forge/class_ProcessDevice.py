import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
class ProcessDevice(BackgroundDevice):
    """A Device that will be run in a background Process.

    See Device for details.
    """
    _launch_class = Process
    context_factory = Context
    'Callable that returns a context. Typically either Context.instance or Context,\n    depending on whether the device should share the global instance or not.\n    '