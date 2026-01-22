import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
class _NotSetType:

    def __repr__(self) -> str:
        return 'NotSet'

    @property
    def value(self) -> Any:
        return None

    @staticmethod
    def remove_unset_items(data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if not isinstance(value, _NotSetType)}