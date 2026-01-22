from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _ClearLine(self):
    self._stream.write('\r{}\r'.format(' ' * self._console_width))