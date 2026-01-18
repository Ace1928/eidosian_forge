import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def vary_by_format():
    return [('bzr', dict(_format='bzr')), ('git', dict(_format='git'))]