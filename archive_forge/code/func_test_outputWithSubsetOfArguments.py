from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_outputWithSubsetOfArguments(self):
    """
        Inputs pass arguments that output will accept.
        """

    class Mechanism(object):
        m = MethodicalMachine()

        @m.input()
        def input(self, x, y=1):
            """an input"""

        @m.state(initial=True)
        def state(self):
            """a state"""

        @m.output()
        def outputX(self, x):
            self._x = x
            return x

        @m.output()
        def outputY(self, y):
            self._y = y
            return y

        @m.output()
        def outputNoArgs(self):
            return None
        state.upon(input, state, [outputX, outputY, outputNoArgs])
    m = Mechanism()
    self.assertEqual(m.input(3), [3, 1, None])
    self.assertEqual(m._x, 3)
    self.assertEqual(m._y, 1)
    self.assertEqual(m.input(x=4), [4, 1, None])
    self.assertEqual(m._x, 4)
    self.assertEqual(m._y, 1)
    self.assertEqual(m.input(6, 3), [6, 3, None])
    self.assertEqual(m._x, 6)
    self.assertEqual(m._y, 3)
    self.assertEqual(m.input(5, y=2), [5, 2, None])
    self.assertEqual(m._x, 5)
    self.assertEqual(m._y, 2)