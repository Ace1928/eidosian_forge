import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def to_dot(nodes):
    lines = ['digraph GraphName {', 'node [shape=rect];', 'rankdir=LR;']
    for i, n in enumerate(nodes):
        lines.append(f'{i} [label={escape(n.label)}, color={('red' if n.root else 'black')}];')
    for i, f in enumerate(nodes):
        for label, j in f.referrents:
            lines.append(f'{i} -> {j} [label = {escape(label)}]')
    lines.append('}\n')
    return '\n'.join(lines)