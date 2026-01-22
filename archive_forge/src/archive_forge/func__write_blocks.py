import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
def _write_blocks(f, prefix, blocks):

    def frames_fragment(frames):
        if not frames:
            return '<non-python>'
        return ';'.join(_frames_fmt(frames, reverse=True))
    for b in blocks:
        if 'history' not in b:
            frames, accounted_for_size = _block_extra(b)
            f.write(f'{prefix};{b['state']};{frames_fragment(frames)} {accounted_for_size}\n')
        else:
            accounted_for_size = 0
            for h in b['history']:
                sz = h['real_size']
                accounted_for_size += sz
                if 'frames' in h:
                    frames = h['frames']
                    f.write(f'{prefix};{b['state']};{frames_fragment(frames)} {sz}\n')
                else:
                    f.write(f'{prefix};{b['state']};<no-context> {sz}\n')
        gaps = b['size'] - accounted_for_size
        if gaps:
            f.write(f'{prefix};{b['state']};<gaps> {gaps}\n')