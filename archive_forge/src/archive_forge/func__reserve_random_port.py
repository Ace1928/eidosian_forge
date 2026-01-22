import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def _reserve_random_port(self, addr: str, *args, **kwargs) -> int:
    with Context() as ctx:
        with ctx.socket(PUSH) as binder:
            for i in range(5):
                port = binder.bind_to_random_port(addr, *args, **kwargs)
                new_addr = '%s:%i' % (addr, port)
                if new_addr in self._random_addrs:
                    continue
                else:
                    break
            else:
                raise ZMQBindError('Could not reserve random port.')
            self._random_addrs.append(new_addr)
    return port