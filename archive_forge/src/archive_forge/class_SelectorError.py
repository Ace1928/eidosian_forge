import errno
import math
import select
import socket
import sys
import time
from collections import namedtuple
from ansible.module_utils.six.moves.collections_abc import Mapping
class SelectorError(Exception):

    def __init__(self, errcode):
        super(SelectorError, self).__init__()
        self.errno = errcode

    def __repr__(self):
        return '<SelectorError errno={0}>'.format(self.errno)

    def __str__(self):
        return self.__repr__()