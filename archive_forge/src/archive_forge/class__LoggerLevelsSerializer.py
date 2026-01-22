from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
class _LoggerLevelsSerializer(object):
    """Serializer for --logger_levels flag."""

    def serialize(self, value):
        if isinstance(value, six.string_types):
            return value
        return ','.join(('{}:{}'.format(name, level) for name, level in value.items()))