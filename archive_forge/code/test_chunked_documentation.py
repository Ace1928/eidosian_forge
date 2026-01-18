import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
Reject chunk markers with no CR character in strict mode.