from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
class AttendanceRecordingIdentityDict(object):
    """A dictionary-like object that records indices of items actually accessed
    from a list."""

    def __init__(self, lst):
        self.l = lst
        self.d = {id(v): i for i, v in enumerate(lst)}
        self.s = set()

    def __getitem__(self, v):
        self.s.add(self.d[id(v)])
        return v