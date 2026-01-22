import unittest
from traits.api import (
from traits.observation.api import (
@observe('foo')
def _do_something_when_foo_changes(self, **kwargs):
    pass