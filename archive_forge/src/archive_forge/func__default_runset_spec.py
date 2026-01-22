from typing import Any, Dict, Optional, TypeVar
from ...public import Api as PublicApi
from ...public import PythonMongoishQueryGenerator, QueryGenerator, Runs
from .util import Attr, Base, coalesce, generate_name, nested_get, nested_set
@staticmethod
def _default_runset_spec():
    return {'id': generate_name(), 'runFeed': {'version': 2, 'columnVisible': {'run:name': False}, 'columnPinned': {}, 'columnWidths': {}, 'columnOrder': [], 'pageSize': 10, 'onlyShowSelected': False}, 'enabled': True, 'selections': {'root': 1, 'bounds': [], 'tree': []}, 'expandedRowAddresses': []}