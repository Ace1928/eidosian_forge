from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _load_single_chat_session(self, cursor: 'sqlite3.Cursor', use_chat_handle_table: bool, chat_id: int) -> ChatSession:
    """
        Load a single chat session from the iMessage chat.db.

        Args:
            cursor: SQLite cursor object.
            chat_id (int): ID of the chat session to load.

        Returns:
            ChatSession: Loaded chat session.
        """
    results: List[HumanMessage] = []
    query = self._get_session_query(use_chat_handle_table)
    cursor.execute(query, (chat_id,))
    messages = cursor.fetchall()
    for date, sender, text, is_from_me, attributedBody in messages:
        if text:
            content = text
        elif attributedBody:
            content = self._parse_attributedBody(attributedBody)
        else:
            continue
        results.append(HumanMessage(role=sender, content=content, additional_kwargs={'message_time': date, 'message_time_as_datetime': nanoseconds_from_2001_to_datetime(date), 'sender': sender, 'is_from_me': bool(is_from_me)}))
    return ChatSession(messages=results)