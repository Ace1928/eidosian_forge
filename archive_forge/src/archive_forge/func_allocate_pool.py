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
@classmethod
@contextmanager
def allocate_pool(cls, num):
    for _ in range(num):
        inst = MinecraftInstance(cls._get_valid_port())
        cls._instance_pool.append(inst)
    yield None
    cls.shutdown()