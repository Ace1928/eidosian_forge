from __future__ import annotations
from os import environ
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase
def test_tzinfo(self) -> None:
    """
        Test that timezone attributes respect the timezone as set by the
        standard C{TZ} environment variable and L{tzset} API.
        """
    if tzset is None:
        raise SkipTest('Platform cannot change timezone; unable to verify offsets.')

    def testForTimeZone(name: str, expectedOffsetDST: str, expectedOffsetSTD: str) -> None:
        setTZ(name)
        localDST = mktime((2006, 6, 30, 0, 0, 0, 4, 181, 1))
        localDSTdt = datetime.fromtimestamp(localDST)
        localSTD = mktime((2007, 1, 31, 0, 0, 0, 2, 31, 0))
        localSTDdt = datetime.fromtimestamp(localSTD)
        tzDST = FixedOffsetTimeZone.fromLocalTimeStamp(localDST)
        tzSTD = FixedOffsetTimeZone.fromLocalTimeStamp(localSTD)
        self.assertEqual(tzDST.tzname(localDSTdt), f'UTC{expectedOffsetDST}')
        self.assertEqual(tzSTD.tzname(localSTDdt), f'UTC{expectedOffsetSTD}')
        self.assertEqual(tzDST.dst(localDSTdt), timedelta(0))
        self.assertEqual(tzSTD.dst(localSTDdt), timedelta(0))

        def timeDeltaFromOffset(offset: str) -> timedelta:
            assert len(offset) == 5
            sign = offset[0]
            hours = int(offset[1:3])
            minutes = int(offset[3:5])
            if sign == '-':
                hours = -hours
                minutes = -minutes
            else:
                assert sign == '+'
            return timedelta(hours=hours, minutes=minutes)
        self.assertEqual(tzDST.utcoffset(localDSTdt), timeDeltaFromOffset(expectedOffsetDST))
        self.assertEqual(tzSTD.utcoffset(localSTDdt), timeDeltaFromOffset(expectedOffsetSTD))
    addTZCleanup(self)
    testForTimeZone('UTC+00', '+0000', '+0000')
    testForTimeZone('EST+05EDT,M4.1.0,M10.5.0', '-0400', '-0500')
    testForTimeZone('CEST-01CEDT,M4.1.0,M10.5.0', '+0200', '+0100')
    testForTimeZone('CST+06', '-0600', '-0600')