import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sqlalchemy import Column, Integer, Text, create_engine
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from sqlalchemy.orm import sessionmaker
def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
    return self.model_class(session_id=session_id, message=json.dumps(message_to_dict(message)))