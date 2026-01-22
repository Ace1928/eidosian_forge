from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class Base20DeprecationWarning(SADeprecationWarning):
    """Issued for usage of APIs specifically deprecated or legacy in
    SQLAlchemy 2.0.

    .. seealso::

        :ref:`error_b8d9`.

        :ref:`deprecation_20_mode`

    """
    deprecated_since: Optional[str] = '1.4'
    'Indicates the version that started raising this deprecation warning'

    def __str__(self) -> str:
        return super().__str__() + ' (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)'