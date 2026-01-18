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
def xlim_type(value):
    try:
        newvalue = [float(x) for x in value.split(',')]
    except:
        raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
    if len(newvalue) != 2:
        raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
    return newvalue