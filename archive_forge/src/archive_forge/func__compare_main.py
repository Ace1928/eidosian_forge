import argparse
import datetime
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import queue
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Tuple, Dict
from . import blas_compare_setup
def _compare_main():
    results = []
    with open(RESULT_FILE, 'rb') as f:
        while True:
            try:
                results.extend(pickle.load(f))
            except EOFError:
                break
    from torch.utils.benchmark import Compare
    comparison = Compare(results)
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()