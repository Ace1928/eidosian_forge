import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
Read an mprofile file and return its content.

    Returns
    =======
    content: dict
        Keys:

        - "mem_usage": (list) memory usage values, in MiB
        - "timestamp": (list) time instant for each memory usage value, in
            second
        - "func_timestamp": (dict) for each function, timestamps and memory
            usage upon entering and exiting.
        - 'cmd_line': (str) command-line ran for this profile.
    