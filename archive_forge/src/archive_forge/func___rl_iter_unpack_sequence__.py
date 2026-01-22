import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_iter_unpack_sequence__(self, it, spec, _getiter_):
    """Protect sequence unpacking of targets in a 'for loop'.

		The target of a for loop could be a sequence.
		For example "for a, b in it"
		=> Each object from the iterator needs guarded sequence unpacking.
		"""
    for ob in _getiter_(it):
        yield self.__rl_unpack_sequence__(ob, spec, _getiter_)