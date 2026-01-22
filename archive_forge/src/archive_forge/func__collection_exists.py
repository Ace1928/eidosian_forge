from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _collection_exists(self) -> bool:
    """Checks whether a collection exists for this message history"""
    try:
        self.client.Collections.get(collection=self.collection)
    except self.rockset.exceptions.NotFoundException:
        return False
    return True