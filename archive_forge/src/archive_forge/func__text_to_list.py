import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def _text_to_list(txt):
    out = txt.strip('][\n').replace("'", '').split(', ')
    return None if out[0] == '' else out