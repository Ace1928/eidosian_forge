import time
import warnings
from typing import Tuple
from zmq import ETERM, POLLERR, POLLIN, POLLOUT, Poller, ZMQError
from .minitornado.ioloop import PeriodicCallback, PollIOLoop
from .minitornado.log import gen_log
@staticmethod
def _map_events(events):
    """translate IOLoop.READ/WRITE/ERROR event masks into zmq.POLLIN/OUT/ERR"""
    z_events = 0
    if events & IOLoop.READ:
        z_events |= POLLIN
    if events & IOLoop.WRITE:
        z_events |= POLLOUT
    if events & IOLoop.ERROR:
        z_events |= POLLERR
    return z_events