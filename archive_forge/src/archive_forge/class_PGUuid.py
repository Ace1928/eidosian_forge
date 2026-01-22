from __future__ import annotations
import datetime as dt
from typing import Any
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from ...sql import sqltypes
from ...sql import type_api
from ...util.typing import Literal
class PGUuid(sqltypes.UUID[sqltypes._UUID_RETURN]):
    render_bind_cast = True
    render_literal_cast = True
    if TYPE_CHECKING:

        @overload
        def __init__(self: PGUuid[_python_UUID], as_uuid: Literal[True]=...) -> None:
            ...

        @overload
        def __init__(self: PGUuid[str], as_uuid: Literal[False]=...) -> None:
            ...

        def __init__(self, as_uuid: bool=True) -> None:
            ...