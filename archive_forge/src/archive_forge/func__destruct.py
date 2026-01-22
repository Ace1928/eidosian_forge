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
def _destruct(self, should_close=False):
    """
        Do our best as the parent process to destruct and kill the child + watcher.
        """
    if (self.running or should_close) and (not self.existing):
        self.running = False
        self._starting = False
        time.sleep(1)
        if self._kill_minecraft_via_malmoenv(self.host, self.port):
            time.sleep(2)
        try:
            minerl.utils.process_watcher.reap_process_and_children(self.minecraft_process)
        except psutil.NoSuchProcess:
            pass
        if self in InstanceManager._instance_pool:
            InstanceManager._instance_pool.remove(self)
            self.release_lock()
    pass