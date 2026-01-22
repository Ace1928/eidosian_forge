from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class OnlyOnePath(object):
    m = MethodicalMachine()

    @m.state(initial=True)
    def start(self):
        """Start state."""

    @m.state()
    def end(self):
        """End state."""

    @m.input()
    def advance(self):
        """Move from start to end."""

    @m.input()
    def deadEnd(self):
        """A transition from nowhere to nowhere."""
    start.upon(advance, end, [])