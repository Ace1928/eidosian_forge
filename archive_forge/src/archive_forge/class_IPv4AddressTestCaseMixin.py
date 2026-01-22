import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IPv4AddressTestCaseMixin(AddressTestCaseMixin):
    addressArgSpec = (('type', '%s'), ('host', '%r'), ('port', '%d'))