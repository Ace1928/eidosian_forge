import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
class TestIntent:

    def test_in_out(self):
        assert str(intent.in_.out) == 'intent(in,out)'
        assert intent.in_.c.is_intent('c')
        assert not intent.in_.c.is_intent_exact('c')
        assert intent.in_.c.is_intent_exact('c', 'in')
        assert intent.in_.c.is_intent_exact('in', 'c')
        assert not intent.in_.is_intent('c')