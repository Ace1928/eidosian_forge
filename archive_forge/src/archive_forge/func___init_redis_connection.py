import warnings
from typing import Any, Dict, List, Optional
from langchain_core._api import deprecated
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
@deprecated('0.0.31', alternative='__init_falkordb_connection')
def __init_redis_connection(self, database: str, host: str='localhost', port: int=6379, username: Optional[str]=None, password: Optional[str]=None, ssl: bool=False) -> None:
    import redis
    from redis.commands.graph import Graph
    warnings.warn('Using the redis package is deprecated. Please use the falkordb package instead, install it with `pip install falkordb`.', DeprecationWarning)
    self._driver = redis.Redis(host=host, port=port, username=username, password=password, ssl=ssl)
    self._graph = Graph(self._driver, database)