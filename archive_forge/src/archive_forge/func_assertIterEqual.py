import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def assertIterEqual(self, iter1, iter2):
    """Check two iterators or iterables are equal independent of order.

        Similar to Python 2.7 assertItemsEqual.  Named differently in order to
        avoid potential conflict.

        Args:
          iter1: An iterator or iterable.
          iter2: An iterator or iterable.
        """
    list1 = list(iter1)
    list2 = list(iter2)
    unmatched1 = list()
    while list1:
        item1 = list1[0]
        del list1[0]
        for index in range(len(list2)):
            if item1 == list2[index]:
                del list2[index]
                break
        else:
            unmatched1.append(item1)
    error_message = []
    for item in unmatched1:
        error_message.append('  Item from iter1 not found in iter2: %r' % item)
    for item in list2:
        error_message.append('  Item from iter2 not found in iter1: %r' % item)
    if error_message:
        self.fail('Collections not equivalent:\n' + '\n'.join(error_message))