from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _reduce_socket(s):
    df = DupFd(s.fileno())
    return (_rebuild_socket, (df, s.family, s.type, s.proto))