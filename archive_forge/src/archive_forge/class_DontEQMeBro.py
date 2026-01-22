from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
class DontEQMeBro:

    def __eq__(this, other):
        self.fail("Don't!")

    def __ne__(this, other):
        self.fail("Don't!")