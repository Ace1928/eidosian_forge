from doctest import ELLIPSIS
from testtools import (
from testtools.assertions import (
from testtools.content import (
from testtools.matchers import (
class TestAssertThatFunction(AssertThatTests, TestCase):

    def assert_that_callable(self, *args, **kwargs):
        return assert_that(*args, **kwargs)