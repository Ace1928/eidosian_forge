import subprocess
from collections import namedtuple
def _giza2pair(pair_string):
    i, j = pair_string.split('-')
    return (int(i), int(j))