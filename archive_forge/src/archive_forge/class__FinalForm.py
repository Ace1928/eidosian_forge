import abc
import collections
import collections.abc
import operator
import sys
import typing
class _FinalForm(typing._SpecialForm, _root=True):

    def __repr__(self):
        return 'typing_extensions.' + self._name

    def __getitem__(self, parameters):
        item = typing._type_check(parameters, f'{self._name} accepts only single type')
        return typing._GenericAlias(self, (item,))