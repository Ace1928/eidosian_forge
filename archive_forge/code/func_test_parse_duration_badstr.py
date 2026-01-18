import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_badstr(self):
    testtuples = ('PPPPPPPPPPPPPPPPPPPPPPPPPPPP', 'PTT', 'PX7DDDTX8888UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU8888888888888888H$H', 'P1Y2M3X.4D', 'P1Y2M3.4XD', 'P1Y2M3DT4H5M6XS', 'PT4H5M6X.2S', 'bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_duration(testtuple, builder=None)