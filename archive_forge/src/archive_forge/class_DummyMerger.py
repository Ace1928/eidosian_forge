import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
class DummyMerger(_mod_merge.ConfigurableFileMerger):
    name_prefix = 'file'