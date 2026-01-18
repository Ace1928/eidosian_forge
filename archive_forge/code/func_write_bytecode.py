import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def write_bytecode(self, install_root):
    """
        Write the `.c` files containing the frozen bytecode.

        Shared frozen modules evenly across the files.
        """
    bytecode_file_names = [f'bytecode_{i}.c' for i in range(NUM_BYTECODE_FILES)]
    bytecode_files = [open(os.path.join(install_root, name), 'w') for name in bytecode_file_names]
    it = itertools.cycle(bytecode_files)
    for m in self.frozen_modules:
        self.write_frozen(m, next(it))
    for f in bytecode_files:
        f.close()