import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def _makeClassAttribute(self, klass: Type[T_gh], value: Any) -> Attribute[T_gh]:
    return GithubObject.__makeTransformedAttribute(value, dict, lambda value: klass(self._requester, self._headers, value, completed=False))