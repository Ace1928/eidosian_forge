import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
Check the changes in a TreeDelta

        This method checks that the TreeDelta contains the expected
        modifications between the two trees that were used to generate
        it. The required changes are passed in as a list, where
        each entry contains the needed information about the change.

        If you do not wish to assert anything about a particular
        category then pass None instead.

        changes: The TreeDelta to check.
        expected_added: a list of (filename,) tuples that must have
            been added in the delta.
        expected_removed: a list of (filename,) tuples that must have
            been removed in the delta.
        expected_modified: a list of (filename,) tuples that must have
            been modified in the delta.
        expected_renamed: a list of (old_path, new_path) tuples that
            must have been renamed in the delta.
        expected_kind_changed: a list of (path, old_kind, new_kind) tuples
            that must have been changed in the delta.
        