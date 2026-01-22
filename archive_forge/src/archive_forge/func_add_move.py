from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def add_move(move):
    """Add an item to six.moves."""
    setattr(_MovedItems, move.name, move)