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
def _kill_minecraft_via_malmoenv(host, port):
    """Use carefully to cause the Minecraft service to exit (and hopefully restart).
        """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((host, port))
        comms.send_message(sock, ('<MalmoEnv' + malmo_version + '/>').encode())
        comms.send_message(sock, '<Exit>NOW</Exit>'.encode())
        reply = comms.recv_message(sock)
        ok, = struct.unpack('!I', reply)
        sock.close()
        return ok == 1
    except Exception as e:
        logger.error('Attempted to send kill command to minecraft process and failed with exception {}'.format(e))
        return False