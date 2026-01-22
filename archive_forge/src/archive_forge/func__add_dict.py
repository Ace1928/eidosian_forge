from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def _add_dict(self, input_dict, name):
    assert name not in self.cache
    assert name not in self.reminders
    assert name not in self.reminder_keys
    self.cache[name] = input_dict
    self.reminders[name] = collections.defaultdict(int)
    self.reminder_keys[name] = collections.defaultdict(set)