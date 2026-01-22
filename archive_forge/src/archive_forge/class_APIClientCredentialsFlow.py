from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.abcs.configs.types import AppEnv
from ..types.properties import StatefulProperty
from ..types.tokens import AccessToken
from ..utils.lazy import get_az_settings, logger
from ..utils.helpers import normalize_audience_name
from typing import List, Optional, Any, Dict, Union, TYPE_CHECKING
class APIClientCredentialsFlow(BaseTokenFlow):
    """
    Can be used to get a token for any API using Client Credentials flow.
    Specify the API with the `audience` param.
    """
    name: Optional[str] = 'api_token'
    schema_type = AccessToken

    def __init__(self, endpoint: str, api_client_id: Optional[str]=None, api_client_env: Optional[Union[str, AppEnv]]=None, audience: Optional[str]=None, client_id: Optional[str]=None, client_secret: Optional[str]=None, oauth_url: Optional[str]=None, **kwargs):
        """
        Initializes the API Client Credentials for the Audience
        """
        key = normalize_audience_name(endpoint)
        if api_client_id:
            key += f'.{api_client_id}'
        if api_client_env:
            if isinstance(api_client_env, AppEnv):
                api_client_env = api_client_env.name
            key += f'.{api_client_env}'
        if audience:
            key += f'.{normalize_audience_name(audience)}'
        super().__init__(audience=audience or self.settings.management_api_url, client_id=client_id or self.settings.client_id, client_secret=client_secret or self.settings.client_secret, oauth_url=oauth_url or self.settings.oauth_url, cache_key=key, **kwargs)
        self.endpoint = endpoint
    '\n    Might need to rework these later\n    '

    def load_data(self) -> Dict[str, Any]:
        """
        Loads the Client Data
        """
        return self.pdict.get(self.data_cache_key, {})

    async def aload_data(self) -> Dict[str, Any]:
        """
        Loads the Client Data
        """
        return await self.pdict.aget(self.data_cache_key, {})

    def save_data(self, data: Dict[str, Any], ex: Optional[int]=None):
        """
        Saves the Data
        """
        self.pdict.set(self.data_cache_key, data, ex=ex)

    async def asave_data(self, data: Dict[str, Any], ex: Optional[int]=None):
        """
        Saves the Data
        """
        await self.pdict.aset(self.data_cache_key, data, ex=ex)