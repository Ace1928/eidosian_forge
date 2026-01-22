from typing import Any, Dict, List, Optional, Sequence
from langchain_community.indexes.base import RecordManager
def _import_motor_asyncio() -> Any:
    """Import Motor if available, otherwise raise error."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        raise ImportError(IMPORT_MOTOR_ASYNCIO_ERROR)
    return AsyncIOMotorClient