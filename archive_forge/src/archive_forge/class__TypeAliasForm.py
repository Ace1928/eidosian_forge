import abc
import collections
import collections.abc
import operator
import sys
import typing
class _TypeAliasForm(typing._SpecialForm, _root=True):

    def __repr__(self):
        return 'typing_extensions.' + self._name