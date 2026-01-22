import os
import pathlib
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Generic, TYPE_CHECKING
from lazyops.types.formatting import to_camel_case, to_snake_case, to_graphql_format
from lazyops.types.classprops import classproperty, lazyproperty
from lazyops.utils.serialization import Json
from pydantic import Field
from pydantic.networks import AnyUrl
from lazyops.imports._pydantic import BaseSettings as _BaseSettings
from lazyops.imports._pydantic import BaseModel as _BaseModel
from lazyops.imports._pydantic import (
class BaseDBUrl(UrlModel):

    @classmethod
    def parse(cls, url: Optional[str]=None, scheme: Optional[str]='http', adapter: Optional[str]=None, **config):
        config_key = cls.__name__.split('DB', 1)[0].lower()
        db = config.get(f'{config_key}_db', config.get('db'))
        path = config.get(f'{config_key}_path', config.get('path'))
        if url:
            if 'http' in scheme and path:
                url += f'/{path}'
            elif db:
                url += f'/{db}'
            return cls(url=url, scheme=scheme, adapter=adapter)
        url = f'{scheme}://'
        host = config.get(f'{config_key}_host', config.get('host'))
        port = config.get(f'{config_key}_port', config.get('port'))
        user = config.get(f'{config_key}_user', config.get('user'))
        password = config.get(f'{config_key}_password', config.get('password'))
        api_key = config.get(f'{config_key}_api_key', config.get('api_key'))
        token = config.get(f'{config_key}_token', config.get('token'))
        if user:
            url += f'{user}'
            if password:
                url += f':{password}'
            url += '@'
        elif api_key:
            url += f'{api_key}@'
        elif token:
            url += f'{token}@'
        url += f'{host}'
        if port:
            url += f':{port}'
        if 'http' in scheme and path:
            url += f'/{path}'
        elif db:
            url += f'/{db}'
        return cls(url=url, scheme=scheme, adapter=adapter)