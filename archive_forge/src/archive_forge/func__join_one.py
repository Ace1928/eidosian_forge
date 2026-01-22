import os
import sys
import time
import traceback
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence
from sphinx.errors import SphinxParallelError
from sphinx.util import logging
def _join_one(self) -> bool:
    joined_any = False
    for tid, pipe in self._precvs.items():
        if pipe.poll():
            exc, logs, result = pipe.recv()
            if exc:
                raise SphinxParallelError(*result)
            for log in logs:
                logger.handle(log)
            self._result_funcs.pop(tid)(self._args.pop(tid), result)
            self._procs[tid].join()
            self._precvs.pop(tid)
            self._pworking -= 1
            joined_any = True
            break
    while self._precvsWaiting and self._pworking < self.nproc:
        newtid, newprecv = self._precvsWaiting.popitem()
        self._precvs[newtid] = newprecv
        self._procs[newtid].start()
        self._pworking += 1
    return joined_any