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
def _run_mc_command(mcdir, *args):
    full_args = ['mc', '-C', mcdir] + list(args)
    with subprocess.Popen(full_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8') as proc:
        retval = proc.wait(10)
        cmd_str = ' '.join(full_args)
        print(f'Cmd: {cmd_str}')
        print(f'  Return: {retval}')
        print(f'  Stdout: {proc.stdout.read()}')
        print(f'  Stderr: {proc.stderr.read()}')
        if retval != 0:
            raise ChildProcessError('Could not run mc')