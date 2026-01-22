from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
class FakeCPUSpec(object):
    """Fake CPU Spec for unit tests."""
    Architecture = mock.sentinel.cpu_arch
    Name = mock.sentinel.cpu_name
    Manufacturer = mock.sentinel.cpu_man
    MaxClockSpeed = mock.sentinel.max_clock_speed
    NumberOfCores = mock.sentinel.cpu_cores
    NumberOfLogicalProcessors = mock.sentinel.cpu_procs