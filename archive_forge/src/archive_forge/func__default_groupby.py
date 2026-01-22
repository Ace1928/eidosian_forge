from typing import Any, Dict, Optional, TypeVar
from ...public import Api as PublicApi
from ...public import PythonMongoishQueryGenerator, QueryGenerator, Runs
from .util import Attr, Base, coalesce, generate_name, nested_get, nested_set
@staticmethod
def _default_groupby():
    return []