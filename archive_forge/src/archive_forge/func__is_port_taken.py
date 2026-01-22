import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
@staticmethod
def _is_port_taken(port, address='0.0.0.0'):
    if psutil.MACOS or psutil.AIX:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((address, port))
            taken = False
        except socket.error as e:
            if e.errno in [98, 10048, 48]:
                taken = True
            else:
                raise e
        s.close()
        return taken
    else:
        ports = [x.laddr.port for x in psutil.net_connections()]
        return port in ports