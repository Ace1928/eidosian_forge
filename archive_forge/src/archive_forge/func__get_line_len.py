import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def _get_line_len(self, n):
    if n not in self.buffer:
        return 0
    line = self.buffer[n]
    if not line:
        return 0
    n = max(line.keys())
    for i in range(n, -1, -1):
        if line[i] != _defchar:
            return i + 1
    return 0