import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def _leak_check():
    current_use = _get_use()
    if current_use - baseline_use > threshold:
        raise Exception('Memory leak detected. Departure from baseline {} after {} iterations'.format(current_use - baseline_use, i))