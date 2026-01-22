import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
class RecordingAddAction(_mod_add.AddAction):

    def __init__(self):
        self.adds = []

    def __call__(self, wt, parent_ie, path, kind):
        self.adds.append((wt, path, kind))