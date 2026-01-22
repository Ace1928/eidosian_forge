from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
class LazyDB(LazyDBBase):

    def __init__(self, dbcache: Any, config: LazyDBConfig):
        super().__init__(dbcache, config)