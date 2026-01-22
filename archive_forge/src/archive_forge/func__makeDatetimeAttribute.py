import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
@staticmethod
def _makeDatetimeAttribute(value: Optional[str]) -> Attribute[datetime]:
    return GithubObject.__makeTransformedAttribute(value, str, _datetime_from_github_isoformat)