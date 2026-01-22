from typing import (
from fastapi._compat import (
from starlette.datastructures import URL as URL  # noqa: F401
from starlette.datastructures import Address as Address  # noqa: F401
from starlette.datastructures import FormData as FormData  # noqa: F401
from starlette.datastructures import Headers as Headers  # noqa: F401
from starlette.datastructures import QueryParams as QueryParams  # noqa: F401
from starlette.datastructures import State as State  # noqa: F401
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
@classmethod
def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
    field_schema.update({'type': 'string', 'format': 'binary'})