import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
class LeapSecondSupportingTestBuilder(BaseTimeBuilder):
    LEAP_SECONDS_SUPPORTED = True