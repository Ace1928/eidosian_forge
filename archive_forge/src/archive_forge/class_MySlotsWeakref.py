import operator
import sys
import types
import unittest
import abc
import pytest
import six
class MySlotsWeakref(object):
    __slots__ = ('__weakref__',)