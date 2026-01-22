import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sqlalchemy import Column, Integer, Text, create_engine
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from sqlalchemy.orm import sessionmaker
class SQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an SQL database."""

    def __init__(self, session_id: str, connection_string: str, table_name: str='message_store', session_id_field_name: str='session_id', custom_message_converter: Optional[BaseMessageConverter]=None):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self.session_id_field_name = session_id_field_name
        self.converter = custom_message_converter or DefaultMessageConverter(table_name)
        self.sql_model_class = self.converter.get_sql_model_class()
        if not hasattr(self.sql_model_class, session_id_field_name):
            raise ValueError('SQL model class must have session_id column')
        self._create_table_if_not_exists()
        self.session_id = session_id
        self.Session = sessionmaker(self.engine)

    def _create_table_if_not_exists(self) -> None:
        self.sql_model_class.metadata.create_all(self.engine)

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages from db"""
        with self.Session() as session:
            result = session.query(self.sql_model_class).where(getattr(self.sql_model_class, self.session_id_field_name) == self.session_id).order_by(self.sql_model_class.id.asc())
            messages = []
            for record in result:
                messages.append(self.converter.from_sql_model(record))
            return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        with self.Session() as session:
            session.add(self.converter.to_sql_model(message, self.session_id))
            session.commit()

    def clear(self) -> None:
        """Clear session memory from db"""
        with self.Session() as session:
            session.query(self.sql_model_class).filter(getattr(self.sql_model_class, self.session_id_field_name) == self.session_id).delete()
            session.commit()