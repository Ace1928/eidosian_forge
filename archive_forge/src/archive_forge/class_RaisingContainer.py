import collections
import enum
import unittest
from traits.trait_base import safe_contains
class RaisingContainer(collections.abc.Sequence):

    def __len__(self):
        return 15

    def __getitem__(self, index):
        if not 0 <= index < 15:
            raise IndexError('Index out of range')
        return 1729

    def __contains__(self, value):
        if value != 1729:
            raise TypeError('My contents are my own private business!')
        return True