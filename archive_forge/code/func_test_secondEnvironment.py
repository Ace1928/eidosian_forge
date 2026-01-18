import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_secondEnvironment(self) -> None:
    """
        L{ListenFDs.fromEnvironment} removes information about the
        inherited file descriptors from the environment mapping so that the
        same inherited file descriptors cannot be handled repeatedly from
        multiple L{ListenFDs} instances.
        """
    env = buildEnvironment(3, os.getpid())
    first = ListenFDs.fromEnvironment(environ=env)
    second = ListenFDs.fromEnvironment(environ=env)
    self.assertEqual(list(range(3, 6)), first.inheritedDescriptors())
    self.assertEqual([], second.inheritedDescriptors())