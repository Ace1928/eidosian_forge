from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_outputsArePrivate(self):
    """
        One of the benefits of using a state machine is that your output method
        implementations don't need to take invalid state transitions into
        account - the methods simply won't be called.  This property would be
        broken if client code called output methods directly, so output methods
        are not directly visible under their names.
        """

    class Machination(object):
        machine = MethodicalMachine()
        counter = 0

        @machine.input()
        def anInput(self):
            """an input"""

        @machine.output()
        def anOutput(self):
            self.counter += 1

        @machine.state(initial=True)
        def state(self):
            """a machine state"""
        state.upon(anInput, enter=state, outputs=[anOutput])
    mach1 = Machination()
    mach1.anInput()
    self.assertEqual(mach1.counter, 1)
    mach2 = Machination()
    with self.assertRaises(AttributeError) as cm:
        mach2.anOutput
    self.assertEqual(mach2.counter, 0)
    self.assertIn('Machination.anOutput is a state-machine output method; to produce this output, call an input method instead.', str(cm.exception))