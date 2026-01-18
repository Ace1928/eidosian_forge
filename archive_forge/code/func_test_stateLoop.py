from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_stateLoop(self):
    """
        It is possible to write a self-loop by omitting "enter"
        """

    class Mechanism(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            """an input"""

        @m.input()
        def say_hi(self):
            """an input"""

        @m.output()
        def _start_say_hi(self):
            return 'hi'

        @m.state(initial=True)
        def start(self):
            """a state"""

        def said_hi(self):
            """a state with no inputs"""
        start.upon(input, outputs=[])
        start.upon(say_hi, outputs=[_start_say_hi])
    a_mechanism = Mechanism()
    [a_greeting] = a_mechanism.say_hi()
    self.assertEqual(a_greeting, 'hi')