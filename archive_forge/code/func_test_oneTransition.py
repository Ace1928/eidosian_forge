from .._core import Automaton, NoTransition
from unittest import TestCase
def test_oneTransition(self):
    """
        L{Automaton.addTransition} adds its input symbol to
        L{Automaton.inputAlphabet}, all its outputs to
        L{Automaton.outputAlphabet}, and causes L{Automaton.outputForInput} to
        start returning the new state and output symbols.
        """
    a = Automaton()
    a.addTransition('beginning', 'begin', 'ending', ['end'])
    self.assertEqual(a.inputAlphabet(), {'begin'})
    self.assertEqual(a.outputAlphabet(), {'end'})
    self.assertEqual(a.outputForInput('beginning', 'begin'), ('ending', ['end']))
    self.assertEqual(a.states(), {'beginning', 'ending'})