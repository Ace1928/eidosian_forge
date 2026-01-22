import collections
import os
import re
import sys
import functools
import itertools
def _get_machine_win32():
    return os.environ.get('PROCESSOR_ARCHITEW6432', '') or os.environ.get('PROCESSOR_ARCHITECTURE', '')