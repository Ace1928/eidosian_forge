import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def write_frozen(self, m: FrozenModule, outfp):
    """Write a single frozen module's bytecode out to a C variable."""
    outfp.write(f'unsigned char {m.c_name}[] = {{')
    for i in range(0, len(m.bytecode), 16):
        outfp.write('\n\t')
        for c in bytes(m.bytecode[i:i + 16]):
            outfp.write('%d,' % c)
    outfp.write('\n};\n')