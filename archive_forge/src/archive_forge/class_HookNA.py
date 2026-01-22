import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
class HookNA(_mod_merge.AbstractPerFileMerger):

    def merge_contents(self, merge_params):
        test.hook_log.append(('no-op',))
        return ('not_applicable', None)