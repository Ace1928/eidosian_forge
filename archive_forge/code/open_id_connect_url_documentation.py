from typing import Optional
from fastapi.openapi.models import OpenIdConnect as OpenIdConnectModel
from fastapi.security.base import SecurityBase
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.status import HTTP_403_FORBIDDEN
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]

    OpenID Connect authentication class. An instance of it would be used as a
    dependency.
    