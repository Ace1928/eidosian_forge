from typing import Any, Dict, List, Optional, Sequence
from langchain_community.indexes.base import RecordManager
def _get_pymongo_client(mongodb_url: str, **kwargs: Any) -> Any:
    """Get MongoClient for sync operations from the mongodb_url,
    otherwise raise error."""
    try:
        pymongo = _import_pymongo()
        client = pymongo(mongodb_url, **kwargs)
    except ValueError as e:
        raise ImportError(f'MongoClient string provided is not in proper format. Got error: {e} ')
    return client