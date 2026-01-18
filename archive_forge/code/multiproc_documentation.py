import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait
from traceback import format_exception
import sys
from logging import INFO
import gc
from copy import deepcopy
import numpy as np
from ... import logging
from ...utils.profiler import get_system_total_memory_gb
from ..engine import MapNode
from .base import DistributedPluginBase
A textwrap.indent replacement for Python < 3.3