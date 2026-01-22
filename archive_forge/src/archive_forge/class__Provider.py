import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
class _Provider(object):
    """A named symbol provider that produces a output at the given index."""

    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __repr__(self):
        if self.name is _TRANSIENT_PROVIDER:
            base = '<TransientProvider'
        else:
            base = "<Provider '%s'" % self.name
        if self.index is None:
            base += '>'
        else:
            base += ' @ index %r>' % self.index
        return base

    def __hash__(self):
        return hash((self.name, self.index))

    def __eq__(self, other):
        return (self.name, self.index) == (other.name, other.index)

    def __ne__(self, other):
        return not self.__eq__(other)