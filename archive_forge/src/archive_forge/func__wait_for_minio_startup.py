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
def _wait_for_minio_startup(mcdir, address, access_key, secret_key):
    start = time.time()
    while time.time() - start < 10:
        try:
            _run_mc_command(mcdir, 'alias', 'set', 'myminio', f'http://{address}', access_key, secret_key)
            return
        except ChildProcessError:
            time.sleep(1)
    raise Exception('mc command could not connect to local minio')