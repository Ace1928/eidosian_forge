from typing import Any, Dict, List, Optional, Sequence
from langchain_community.indexes.base import RecordManager
def _import_pymongo() -> Any:
    """Import PyMongo if available, otherwise raise error."""
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError(IMPORT_PYMONGO_ERROR)
    return MongoClient