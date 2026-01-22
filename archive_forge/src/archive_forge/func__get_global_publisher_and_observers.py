import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
def _get_global_publisher_and_observers():
    """Return ``(log_publisher, observers)``.

    Twisted 15.2.0 changed the logging framework. This method will always
    return a tuple of the global log publisher and all observers associated
    with that publisher.
    """
    if globalLogPublisher is not None:
        publisher = globalLogPublisher
        return (publisher, list(publisher._observers))
    else:
        publisher = log.theLogPublisher
        return (publisher, list(publisher.observers))