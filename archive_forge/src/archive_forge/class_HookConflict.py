import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
class HookConflict(_mod_merge.AbstractPerFileMerger):

    def merge_contents(self, merge_params):
        test.hook_log.append(('conflict',))
        if merge_params.this_path == 'name1':
            return ('conflicted', [b'text-with-conflict-markers-from-hook'])
        return ('not_applicable', None)