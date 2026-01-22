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
def _get_line(self, n):
    line = self.buffer[n]
    line_len = self._get_line_len(n)
    out = [line[i].data for i in range(line_len)]
    idxs = np.arange(line_len)
    insert = lambda i, c: (out.insert(idxs[i], c), idxs[i:].__iadd__(1))
    fgs = [int(_defchar.fg)] + [int(line[i].fg) for i in range(line_len)]
    [insert(i, _get_char(line[int(i)].fg)) for i in np.where(np.diff(fgs))[0]] if len(set(fgs)) > 1 else None
    bgs = [int(_defchar.bg)] + [int(line[i].bg) for i in range(line_len)]
    [insert(i, _get_char(line[int(i)].bg)) for i in np.where(np.diff(bgs))[0]] if len(set(bgs)) > 1 else None
    attrs = {k: [False] + [line[i][k] for i in range(line_len)] for k in Char.__slots__[3:]}
    [[insert(i, _get_char(ANSI_STYLES_REV[k if line[int(i)][k] else '/' + k])) for i in np.where(np.diff(v))[0]] for k, v in attrs.items() if any(v)]
    return ''.join(out)