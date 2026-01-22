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
class RedisDB(BaseDBUrl):
    url: AnyUrl
    scheme: Optional[str] = 'redis'
    adapter: Optional[str] = None

    @classmethod
    def parse(cls, url: Optional[str]=None, scheme: Optional[str]='redis', adapter: Optional[str]=None, **config):
        return super().parse(url=url, scheme=scheme, adapter=adapter, **config)