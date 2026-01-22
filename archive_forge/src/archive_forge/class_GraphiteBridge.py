import logging
import re
import socket
import threading
import time
from timeit import default_timer
from typing import Callable, Tuple
from ..registry import CollectorRegistry, REGISTRY
class GraphiteBridge:

    def __init__(self, address: Tuple[str, int], registry: CollectorRegistry=REGISTRY, timeout_seconds: float=30, _timer: Callable[[], float]=time.time, tags: bool=False):
        self._address = address
        self._registry = registry
        self._tags = tags
        self._timeout = timeout_seconds
        self._timer = _timer

    def push(self, prefix: str='') -> None:
        now = int(self._timer())
        output = []
        prefixstr = ''
        if prefix:
            prefixstr = prefix + '.'
        for metric in self._registry.collect():
            for s in metric.samples:
                if s.labels:
                    if self._tags:
                        sep = ';'
                        fmt = '{0}={1}'
                    else:
                        sep = '.'
                        fmt = '{0}.{1}'
                    labelstr = sep + sep.join([fmt.format(_sanitize(k), _sanitize(v)) for k, v in sorted(s.labels.items())])
                else:
                    labelstr = ''
                output.append(f'{prefixstr}{_sanitize(s.name)}{labelstr} {float(s.value)} {now}\n')
        conn = socket.create_connection(self._address, self._timeout)
        conn.sendall(''.join(output).encode('ascii'))
        conn.close()

    def start(self, interval: float=60.0, prefix: str='') -> None:
        t = _RegularPush(self, interval, prefix)
        t.daemon = True
        t.start()