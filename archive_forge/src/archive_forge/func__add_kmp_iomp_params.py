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
def _add_kmp_iomp_params(parser):
    group = parser.add_argument_group('IOMP Parameters')
    group.add_argument('--disable-iomp', '--disable_iomp', action='store_true', default=False, help='By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD')