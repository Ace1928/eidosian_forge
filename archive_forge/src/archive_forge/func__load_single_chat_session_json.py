import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, List, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _load_single_chat_session_json(self, file_path: str) -> ChatSession:
    """Load a single chat session from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            ChatSession: The loaded chat session.
        """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    messages = data.get('messages', [])
    results: List[BaseMessage] = []
    for message in messages:
        text = message.get('text', '')
        timestamp = message.get('date', '')
        from_name = message.get('from', '')
        results.append(HumanMessage(content=text, additional_kwargs={'sender': from_name, 'events': [{'message_time': timestamp}]}))
    return ChatSession(messages=results)