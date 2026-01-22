import warnings
from typing import Any, Dict, List, Optional
from langchain_core._api import deprecated
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
def __init_falkordb_connection(self, database: str, host: str='localhost', port: int=6379, username: Optional[str]=None, password: Optional[str]=None, ssl: bool=False) -> None:
    from falkordb import FalkorDB
    try:
        self._driver = FalkorDB(host=host, port=port, username=username, password=password, ssl=ssl)
    except Exception as e:
        raise ConnectionError(f'Failed to connect to FalkorDB: {e}')
    self._graph = self._driver.select_graph(database)