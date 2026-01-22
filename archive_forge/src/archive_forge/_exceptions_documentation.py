from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload
A combination of multiple unrelated exceptions.