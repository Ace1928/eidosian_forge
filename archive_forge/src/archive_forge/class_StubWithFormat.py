from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
class StubWithFormat:
    """A stub object used to check that convenience methods call Inter's."""
    _format = object()