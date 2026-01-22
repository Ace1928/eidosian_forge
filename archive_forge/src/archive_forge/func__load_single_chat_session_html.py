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
def _load_single_chat_session_html(self, file_path: str) -> ChatSession:
    """Load a single chat session from an HTML file.

        Args:
            file_path (str): Path to the HTML file.

        Returns:
            ChatSession: The loaded chat session.
        """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Please install the 'beautifulsoup4' package to load Telegram HTML files. You can do this by running'pip install beautifulsoup4' in your terminal.")
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    results: List[Union[HumanMessage, AIMessage]] = []
    previous_sender = None
    for message in soup.select('.message.default'):
        timestamp = message.select_one('.pull_right.date.details')['title']
        from_name_element = message.select_one('.from_name')
        if from_name_element is None and previous_sender is None:
            logger.debug('from_name not found in message')
            continue
        elif from_name_element is None:
            from_name = previous_sender
        else:
            from_name = from_name_element.text.strip()
        text = message.select_one('.text').text.strip()
        results.append(HumanMessage(content=text, additional_kwargs={'sender': from_name, 'events': [{'message_time': timestamp}]}))
        previous_sender = from_name
    return ChatSession(messages=results)