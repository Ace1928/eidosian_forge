import os
import argparse
import logging
import shutil
import multiprocessing as mp
from contextlib import closing
from functools import partial
import fontTools
from .ufo import font_to_quadratic, fonts_to_quadratic
def _cpu_count():
    try:
        return mp.cpu_count()
    except NotImplementedError:
        return 1