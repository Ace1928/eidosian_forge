import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def add_lib_preload(self, lib_type):
    """Enable TCMalloc/JeMalloc/intel OpenMP."""
    library_paths = []
    if 'CONDA_PREFIX' in os.environ:
        library_paths.append(f'{os.environ['CONDA_PREFIX']}/lib')
    if 'VIRTUAL_ENV' in os.environ:
        library_paths.append(f'{os.environ['VIRTUAL_ENV']}/lib')
    library_paths += [f'{expanduser('~')}/.local/lib', '/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64']
    lib_find = False
    lib_set = False
    for item in os.getenv('LD_PRELOAD', '').split(':'):
        if item.endswith(f'lib{lib_type}.so'):
            lib_set = True
            break
    if not lib_set:
        for lib_path in library_paths:
            library_file = os.path.join(lib_path, f'lib{lib_type}.so')
            matches = glob.glob(library_file)
            if len(matches) > 0:
                ld_preloads = [f'{matches[0]}', os.getenv('LD_PRELOAD', '')]
                os.environ['LD_PRELOAD'] = os.pathsep.join([p.strip(os.pathsep) for p in ld_preloads if p])
                lib_find = True
                break
    return lib_set or lib_find