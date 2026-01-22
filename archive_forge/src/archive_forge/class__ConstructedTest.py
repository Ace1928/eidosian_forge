from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
class _ConstructedTest(TestCase):
    """A test case defined by arguments, rather than overrides."""

    def __init__(self, test_method_name, set_up, test_body, tear_down, cleanups, pre_set_up, post_tear_down):
        """Construct a test case.

        See ``make_test_case`` for full documentation.
        """
        setattr(self, test_method_name, self.test_case)
        super().__init__(test_method_name)
        self._set_up = set_up
        self._test_body = test_body
        self._tear_down = tear_down
        self._test_cleanups = cleanups
        self._pre_set_up = pre_set_up
        self._post_tear_down = post_tear_down

    def setUp(self):
        self._pre_set_up(self)
        super().setUp()
        for cleanup in self._test_cleanups:
            self.addCleanup(cleanup, self)
        self._set_up(self)

    def test_case(self):
        self._test_body(self)

    def tearDown(self):
        self._tear_down(self)
        super().tearDown()
        self._post_tear_down(self)