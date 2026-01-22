import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_unpack_sequence__(self, it, spec, _getiter_):
    """Protect nested sequence unpacking.

		Protect the unpacking of 'it' by wrapping it with '_getiter_'.
		Furthermore for each child element, defined by spec,
		__rl_unpack_sequence__ is called again.

		Have a look at transformer.py 'gen_unpack_spec' for a more detailed
		explanation.
		"""
    ret = list(self.__rl__getiter__(it))
    if len(ret) < spec['min_len']:
        return ret
    for idx, child_spec in spec['childs']:
        ret[idx] = self.__rl_unpack_sequence__(ret[idx], child_spec, _getiter_)
    return ret