from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _rebuild_socket(df, family, type, proto):
    fd = df.detach()
    return socket.socket(family, type, proto, fileno=fd)