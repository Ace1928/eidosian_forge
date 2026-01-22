import dill
from pydantic import BaseModel, Field, root_validator
from pydantic.types import ByteSize
from typing import Union, Any, Dict, Optional
from lazyops.configs.base import DefaultSettings
from lazyops.types import lazyproperty
from lazyops.libs.sqlcache.constants import DEFAULT_SETTINGS, OPTIMIZED_SETTINGS, DBNAME
class SqlCacheSettings(DefaultSettings):
    """
    Settings for the SqlCache.
    """

    class Config(DefaultSettings.Config):
        env_prefix = 'SQLCACHE_'
        case_sensitive = False