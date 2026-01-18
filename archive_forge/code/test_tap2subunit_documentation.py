from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
Tests for TAP2SubUnit.

    These tests test TAP string data in, and subunit string data out.
    This is ok because the subunit protocol is intended to be stable,
    but it might be easier/pithier to write tests against TAP string in,
    parsed subunit objects out (by hooking the subunit stream to a subunit
    protocol server.
    