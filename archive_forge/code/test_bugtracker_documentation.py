from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
When given a bug identifier that is invalid for Trac, get_bug_url
        should raise an error.
        