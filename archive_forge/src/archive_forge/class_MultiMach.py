from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class MultiMach(object):
    a = MethodicalMachine()
    b = MethodicalMachine()

    @a.input()
    def inputA(self):
        """input A"""

    @b.input()
    def inputB(self):
        """input B"""

    @a.state(initial=True)
    def initialA(self):
        """initial A"""

    @b.state(initial=True)
    def initialB(self):
        """initial B"""

    @a.output()
    def outputA(self):
        return 'A'

    @b.output()
    def outputB(self):
        return 'B'
    initialA.upon(inputA, initialA, [outputA])
    initialB.upon(inputB, initialB, [outputB])