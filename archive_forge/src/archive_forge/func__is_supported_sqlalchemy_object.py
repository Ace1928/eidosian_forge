import math
import numpy as np
import pandas
from modin.config import NPartitions, ReadSqlEngine
from modin.core.io.file_dispatcher import FileDispatcher
from modin.db_conn import ModinDatabaseConnection
@classmethod
def _is_supported_sqlalchemy_object(cls, obj):
    supported = None
    try:
        import sqlalchemy as sa
        supported = isinstance(obj, (sa.engine.Engine, sa.engine.Connection))
    except ImportError:
        supported = False
    return supported