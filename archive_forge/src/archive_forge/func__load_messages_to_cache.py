import json
import logging
from datetime import datetime
from typing import List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
def _load_messages_to_cache(self) -> None:
    """
        Loads messages from the database into the cache.

        This method retrieves messages from the database table. The retrieved messages
        are then stored in the cache for faster access.

        Raises:
            SQLAlchemyError: If there is an error executing the database query.

        """
    time_condition = f"AND create_time >= '{self.earliest_time}'" if self.earliest_time else ''
    query = text(f'\n            SELECT message FROM {self.table_name} \n            WHERE session_id = :session_id {time_condition} \n            ORDER BY id;\n        ')
    try:
        result = self.session.execute(query, {'session_id': self.session_id})
        for record in result.fetchall():
            message_dict = json.loads(record[0])
            self.cache.append(messages_from_dict([message_dict])[0])
    except SQLAlchemyError as e:
        logger.error(f'Error loading messages to cache: {e}')