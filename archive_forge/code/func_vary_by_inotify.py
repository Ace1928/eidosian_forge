import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def vary_by_inotify():
    return [('with_inotify', dict(_use_inotify=True)), ('without_inotify', dict(_use_inotify=False))]