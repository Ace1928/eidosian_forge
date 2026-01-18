import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames

        Any subsequent calls to C{inheritedDescriptors} and
        C{inheritedNamedDescriptors} return the same list.
        