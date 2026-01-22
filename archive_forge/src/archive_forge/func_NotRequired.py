import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@_ExtensionsSpecialForm
def NotRequired(self, parameters):
    """A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """
    item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
    return typing._GenericAlias(self, (item,))