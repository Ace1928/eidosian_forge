import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def _makeListOfClassesAttribute(self, klass: Type[T_gh], value: Any) -> Attribute[List[T_gh]]:
    if isinstance(value, list) and all((isinstance(element, dict) for element in value)):
        return _ValuedAttribute([klass(self._requester, self._headers, element, completed=False) for element in value])
    else:
        return _BadAttribute(value, [dict])