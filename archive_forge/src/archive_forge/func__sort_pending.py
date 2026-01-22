import argparse
import logging
import socket
import struct
import sys
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Union
def _sort_pending(self, pending: List[WorkerEntry]) -> List[WorkerEntry]:
    if self._sortby == 'host':
        pending.sort(key=lambda s: s.host)
    elif self._sortby == 'task':
        pending.sort(key=lambda s: s.task_id)
    return pending