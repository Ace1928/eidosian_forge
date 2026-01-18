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
A special typing construct to mark an item of a TypedDict as read-only.

        For example:

            class Movie(TypedDict):
                title: ReadOnly[str]
                year: int

            def mutate_movie(m: Movie) -> None:
                m["year"] = 1992  # allowed
                m["title"] = "The Matrix"  # typechecker error

        There is no runtime checking for this property.
        