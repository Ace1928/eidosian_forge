from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _document_exists(self) -> bool:
    return len(self._query(f'\n                        SELECT 1\n                        FROM {self.location} \n                        WHERE _id=:session_id\n                        LIMIT 1\n                    ', session_id=self.session_id)) != 0