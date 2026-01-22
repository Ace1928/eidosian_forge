from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _reduce_method(m):
    if m.__self__ is None:
        return (getattr, (m.__class__, m.__func__.__name__))
    else:
        return (getattr, (m.__self__, m.__func__.__name__))