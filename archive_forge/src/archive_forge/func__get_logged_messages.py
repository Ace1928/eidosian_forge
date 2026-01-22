import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def _get_logged_messages(self, function, *args, **kwargs):
    """Run ``function`` and return ``(ret, logged_messages)``."""
    messages = []
    publisher, _ = _get_global_publisher_and_observers()
    publisher.addObserver(messages.append)
    try:
        ret = function(*args, **kwargs)
    finally:
        publisher.removeObserver(messages.append)
    return (ret, messages)