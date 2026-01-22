import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def _storeAndUseAttributes(self, headers: Dict[str, Union[str, int]], attributes: Any) -> None:
    self._headers = headers
    self._rawData = attributes
    self._useAttributes(attributes)