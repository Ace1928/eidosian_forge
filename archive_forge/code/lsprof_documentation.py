import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
Save profiling data to a file.

        :param filename: the name of the output file
        :param format: 'txt' for a text representation;
            'callgrind' for calltree format;
            otherwise a pickled Python object. A format of None indicates
            that the format to use is to be found from the filename. If
            the name starts with callgrind.out, callgrind format is used
            otherwise the format is given by the filename extension.
        