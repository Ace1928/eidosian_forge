from typing import Any, Dict, List, Optional, Union, cast
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import OAuth2 as OAuth2Model
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.param_functions import Form
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
class OAuth2PasswordBearer(OAuth2):
    """
    OAuth2 flow for authentication using a bearer token obtained with a password.
    An instance of it would be used as a dependency.

    Read more about it in the
    [FastAPI docs for Simple OAuth2 with Password and Bearer](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/).
    """

    def __init__(self, tokenUrl: Annotated[str, Doc('\n                The URL to obtain the OAuth2 token. This would be the *path operation*\n                that has `OAuth2PasswordRequestForm` as a dependency.\n                ')], scheme_name: Annotated[Optional[str], Doc('\n                Security scheme name.\n\n                It will be included in the generated OpenAPI (e.g. visible at `/docs`).\n                ')]=None, scopes: Annotated[Optional[Dict[str, str]], Doc('\n                The OAuth2 scopes that would be required by the *path operations* that\n                use this dependency.\n                ')]=None, description: Annotated[Optional[str], Doc('\n                Security scheme description.\n\n                It will be included in the generated OpenAPI (e.g. visible at `/docs`).\n                ')]=None, auto_error: Annotated[bool, Doc('\n                By default, if no HTTP Auhtorization header is provided, required for\n                OAuth2 authentication, it will automatically cancel the request and\n                send the client an error.\n\n                If `auto_error` is set to `False`, when the HTTP Authorization header\n                is not available, instead of erroring out, the dependency result will\n                be `None`.\n\n                This is useful when you want to have optional authentication.\n\n                It is also useful when you want to have authentication that can be\n                provided in one of multiple optional ways (for example, with OAuth2\n                or in a cookie).\n                ')]=True):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password=cast(Any, {'tokenUrl': tokenUrl, 'scopes': scopes}))
        super().__init__(flows=flows, scheme_name=scheme_name, description=description, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail='Not authenticated', headers={'WWW-Authenticate': 'Bearer'})
            else:
                return None
        return param