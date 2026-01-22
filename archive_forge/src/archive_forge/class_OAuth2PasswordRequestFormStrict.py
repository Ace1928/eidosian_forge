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
class OAuth2PasswordRequestFormStrict(OAuth2PasswordRequestForm):
    """
    This is a dependency class to collect the `username` and `password` as form data
    for an OAuth2 password flow.

    The OAuth2 specification dictates that for a password flow the data should be
    collected using form data (instead of JSON) and that it should have the specific
    fields `username` and `password`.

    All the initialization parameters are extracted from the request.

    The only difference between `OAuth2PasswordRequestFormStrict` and
    `OAuth2PasswordRequestForm` is that `OAuth2PasswordRequestFormStrict` requires the
    client to send the form field `grant_type` with the value `"password"`, which
    is required in the OAuth2 specification (it seems that for no particular reason),
    while for `OAuth2PasswordRequestForm` `grant_type` is optional.

    Read more about it in the
    [FastAPI docs for Simple OAuth2 with Password and Bearer](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/).

    ## Example

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI
    from fastapi.security import OAuth2PasswordRequestForm

    app = FastAPI()


    @app.post("/login")
    def login(form_data: Annotated[OAuth2PasswordRequestFormStrict, Depends()]):
        data = {}
        data["scopes"] = []
        for scope in form_data.scopes:
            data["scopes"].append(scope)
        if form_data.client_id:
            data["client_id"] = form_data.client_id
        if form_data.client_secret:
            data["client_secret"] = form_data.client_secret
        return data
    ```

    Note that for OAuth2 the scope `items:read` is a single scope in an opaque string.
    You could have custom internal logic to separate it by colon caracters (`:`) or
    similar, and get the two parts `items` and `read`. Many applications do that to
    group and organize permisions, you could do it as well in your application, just
    know that that it is application specific, it's not part of the specification.


    grant_type: the OAuth2 spec says it is required and MUST be the fixed string "password".
        This dependency is strict about it. If you want to be permissive, use instead the
        OAuth2PasswordRequestForm dependency class.
    username: username string. The OAuth2 spec requires the exact field name "username".
    password: password string. The OAuth2 spec requires the exact field name "password".
    scope: Optional string. Several scopes (each one a string) separated by spaces. E.g.
        "items:read items:write users:read profile openid"
    client_id: optional string. OAuth2 recommends sending the client_id and client_secret (if any)
        using HTTP Basic auth, as: client_id:client_secret
    client_secret: optional string. OAuth2 recommends sending the client_id and client_secret (if any)
        using HTTP Basic auth, as: client_id:client_secret
    """

    def __init__(self, grant_type: Annotated[str, Form(pattern='password'), Doc('\n                The OAuth2 spec says it is required and MUST be the fixed string\n                "password". This dependency is strict about it. If you want to be\n                permissive, use instead the `OAuth2PasswordRequestForm` dependency\n                class.\n                ')], username: Annotated[str, Form(), Doc('\n                `username` string. The OAuth2 spec requires the exact field name\n                `username`.\n                ')], password: Annotated[str, Form(), Doc('\n                `password` string. The OAuth2 spec requires the exact field name\n                `password".\n                ')], scope: Annotated[str, Form(), Doc('\n                A single string with actually several scopes separated by spaces. Each\n                scope is also a string.\n\n                For example, a single string with:\n\n                ```python\n                "items:read items:write users:read profile openid"\n                ````\n\n                would represent the scopes:\n\n                * `items:read`\n                * `items:write`\n                * `users:read`\n                * `profile`\n                * `openid`\n                ')]='', client_id: Annotated[Union[str, None], Form(), Doc("\n                If there's a `client_id`, it can be sent as part of the form fields.\n                But the OAuth2 specification recommends sending the `client_id` and\n                `client_secret` (if any) using HTTP Basic auth.\n                ")]=None, client_secret: Annotated[Union[str, None], Form(), Doc("\n                If there's a `client_password` (and a `client_id`), they can be sent\n                as part of the form fields. But the OAuth2 specification recommends\n                sending the `client_id` and `client_secret` (if any) using HTTP Basic\n                auth.\n                ")]=None):
        super().__init__(grant_type=grant_type, username=username, password=password, scope=scope, client_id=client_id, client_secret=client_secret)