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
def _makeDictAttribute(value: Dict[str, Any]) -> Attribute[Dict[str, Any]]:
    return GithubObject.__makeSimpleAttribute(value, dict)