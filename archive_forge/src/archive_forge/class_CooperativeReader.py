import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
class CooperativeReader(object):
    """
    An eventlet thread friendly class for reading in image data.

    When accessing data either through the iterator or the read method
    we perform a sleep to allow a co-operative yield. When there is more than
    one image being uploaded/downloaded this prevents eventlet thread
    starvation, ie allows all threads to be scheduled periodically rather than
    having the same thread be continuously active.
    """

    def __init__(self, fd):
        """
        :param fd: Underlying image file object
        """
        self.fd = fd
        self.iterator = None
        if hasattr(fd, 'read'):
            self.read = cooperative_read(fd)
        else:
            self.iterator = None
            self.buffer = b''
            self.position = 0

    def read(self, length=None):
        """Return the requested amount of bytes, fetching the next chunk of
        the underlying iterator when needed.

        This is replaced with cooperative_read in __init__ if the underlying
        fd already supports read().
        """
        if length is None:
            if len(self.buffer) - self.position > 0:
                result = self.buffer[self.position:]
                self.buffer = b''
                self.position = 0
                return bytes(result)
            else:
                try:
                    if self.iterator is None:
                        self.iterator = self.__iter__()
                    return next(self.iterator)
                except StopIteration:
                    return b''
                finally:
                    self.buffer = b''
                    self.position = 0
        else:
            result = bytearray()
            while len(result) < length:
                if self.position < len(self.buffer):
                    to_read = length - len(result)
                    chunk = self.buffer[self.position:self.position + to_read]
                    result.extend(chunk)
                    if len(result) >= MAX_COOP_READER_BUFFER_SIZE:
                        raise exception.LimitExceeded()
                    self.position += len(chunk)
                else:
                    try:
                        if self.iterator is None:
                            self.iterator = self.__iter__()
                        self.buffer = next(self.iterator)
                        self.position = 0
                    except StopIteration:
                        self.buffer = b''
                        self.position = 0
                        return bytes(result)
            return bytes(result)

    def __iter__(self):
        return cooperative_iter(self.fd.__iter__())