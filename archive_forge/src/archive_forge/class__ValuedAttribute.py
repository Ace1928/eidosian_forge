import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
class _ValuedAttribute(Attribute[T]):

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value