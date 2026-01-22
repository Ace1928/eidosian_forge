from typing import Callable
import zmq
from zmq.backend import monitored_queue as _backend_mq
def _relay(ins, outs, sides, prefix, swap_ids):
    msg = ins.recv_multipart()
    if swap_ids:
        msg[:2] = msg[:2][::-1]
    outs.send_multipart(msg)
    sides.send_multipart([prefix] + msg)