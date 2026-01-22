from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
class LazyListWrapper(collections_abc.MutableSequence):
    """Wraps a list that does not exist at object creation time.

  We sometimes have a need to allow access to a list property of a nested
  message, when we're not sure if all the layers above the list exist yet.
  We want to arrange it so that when you write to the list, all the above
  messages are lazily created.

  When you create a LazyListWrapper, you pass in a create function, which
  must do whatever setup you need to do, and then return the list that it
  creates in an underlying message.

  As soon as you start adding items to the LazyListWrapper, it will do the
  setup for you. Until then, it won't create any underlying messages.
  """

    def __init__(self, create):
        self._create = create
        self._l = None

    def __getitem__(self, i):
        if self._l:
            return self._l[i]
        raise IndexError()

    def __setitem__(self, i, v):
        if self._l is None:
            self._l = self._create()
        self._l[i] = v

    def __delitem__(self, i):
        if self._l:
            del self._l[i]
        else:
            raise IndexError()

    def __len__(self):
        if self._l:
            return len(self._l)
        return 0

    def insert(self, i, v):
        if self._l is None:
            self._l = self._create()
        self._l.insert(i, v)