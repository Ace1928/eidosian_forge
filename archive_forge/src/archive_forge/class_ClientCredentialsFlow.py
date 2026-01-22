from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.abcs.configs.types import AppEnv
from ..types.properties import StatefulProperty
from ..types.tokens import AccessToken
from ..utils.lazy import get_az_settings, logger
from ..utils.helpers import normalize_audience_name
from typing import List, Optional, Any, Dict, Union, TYPE_CHECKING
class ClientCredentialsFlow(BaseTokenFlow):
    """
    Can be used to get a token for any API using Client Credentials flow.
    Specify the API with the `audience` param.
    """
    name: Optional[str] = 'client_token'
    schema_type = AccessToken