import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestParametrizedResolveConflicts(tests.TestCaseWithTransport):
    """This class provides a base to test single conflict resolution.

    Since all conflict objects are created with specific semantics for their
    attributes, each class should implement the necessary functions and
    attributes described below.

    Each class should define the scenarios that create the expected (single)
    conflict.

    Each scenario describes:
    * how to create 'base' tree (and revision)
    * how to create 'left' tree (and revision, parent rev 'base')
    * how to create 'right' tree (and revision, parent rev 'base')
    * how to check that changes in 'base'->'left' have been taken
    * how to check that changes in 'base'->'right' have been taken

    From each base scenario, we generate two concrete scenarios where:
    * this=left, other=right
    * this=right, other=left

    Then the test case verifies each concrete scenario by:
    * creating a branch containing the 'base', 'this' and 'other' revisions
    * creating a working tree for the 'this' revision
    * performing the merge of 'other' into 'this'
    * verifying the expected conflict was generated
    * resolving with --take-this or --take-other, and running the corresponding
      checks (for either 'base'->'this', or 'base'->'other')

    :cvar _conflict_type: The expected class of the generated conflict.

    :cvar _assert_conflict: A method receiving the working tree and the
        conflict object and checking its attributes.

    :cvar _base_actions: The branchbuilder actions to create the 'base'
        revision.

    :cvar _this: The dict related to 'base' -> 'this'. It contains at least:
      * 'actions': The branchbuilder actions to create the 'this'
          revision.
      * 'check': how to check the changes after resolution with --take-this.

    :cvar _other: The dict related to 'base' -> 'other'. It contains at least:
      * 'actions': The branchbuilder actions to create the 'other'
          revision.
      * 'check': how to check the changes after resolution with --take-other.
    """
    _conflict_type: Type[conflicts.Conflict]
    _assert_conflict: Callable[[Any, Any, Any], Any]
    _base_actions = None
    _this = None
    _other = None
    scenarios: List[Tuple[Dict[str, Any], Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = []
    "The scenario list for the conflict type defined by the class.\n\n    Each scenario is of the form:\n    (common, (left_name, left_dict), (right_name, right_dict))\n\n    * common is a dict\n\n    * left_name and right_name are the scenario names that will be combined\n\n    * left_dict and right_dict are the attributes specific to each half of\n      the scenario. They should include at least 'actions' and 'check' and\n      will be available as '_this' and '_other' test instance attributes.\n\n    Daughters classes are free to add their specific attributes as they see\n    fit in any of the three dicts.\n\n    This is a class method so that load_tests can find it.\n\n    '_base_actions' in the common dict, 'actions' and 'check' in the left\n    and right dicts use names that map to methods in the test classes. Some\n    prefixes are added to these names to get the correspong methods (see\n    _get_actions() and _get_check()). The motivation here is to avoid\n    collisions in the class namespace.\n    "

    def setUp(self):
        super().setUp()
        builder = self.make_branch_builder('trunk')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'start')
        base_actions = self._get_actions(self._base_actions)()
        builder.build_snapshot([b'start'], base_actions, revision_id=b'base')
        actions_other = self._get_actions(self._other['actions'])()
        builder.build_snapshot([b'base'], actions_other, revision_id=b'other')
        actions_this = self._get_actions(self._this['actions'])()
        builder.build_snapshot([b'base'], actions_this, revision_id=b'this')
        builder.finish_series()
        self.builder = builder

    def _get_actions(self, name):
        return getattr(self, 'do_%s' % name)

    def _get_check(self, name):
        return getattr(self, 'check_%s' % name)

    def _merge_other_into_this(self):
        b = self.builder.get_branch()
        wt = b.controldir.sprout('branch').open_workingtree()
        wt.merge_from_branch(b, b'other')
        return wt

    def assertConflict(self, wt):
        confs = wt.conflicts()
        self.assertLength(1, confs)
        c = confs[0]
        self.assertIsInstance(c, self._conflict_type)
        self._assert_conflict(wt, c)

    def _get_resolve_path_arg(self, wt, action):
        raise NotImplementedError(self._get_resolve_path_arg)

    def check_resolved(self, wt, action):
        path = self._get_resolve_path_arg(wt, action)
        conflicts.resolve(wt, [path], action=action)
        self.assertLength(0, wt.conflicts())
        self.assertLength(0, list(wt.unknowns()))

    def test_resolve_taking_this(self):
        wt = self._merge_other_into_this()
        self.assertConflict(wt)
        self.check_resolved(wt, 'take_this')
        check_this = self._get_check(self._this['check'])
        check_this()

    def test_resolve_taking_other(self):
        wt = self._merge_other_into_this()
        self.assertConflict(wt)
        self.check_resolved(wt, 'take_other')
        check_other = self._get_check(self._other['check'])
        check_other()