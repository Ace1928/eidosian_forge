from typing import Callable
import zmq
from zmq.backend import monitored_queue as _backend_mq
pure Python monitored_queue function

For use when Cython extension is unavailable (PyPy).

Authors
-------
* MinRK
