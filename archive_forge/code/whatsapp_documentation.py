import logging
import os
import re
import zipfile
from typing import Iterator, List, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
Lazy load the messages from the chat file and yield
        them as chat sessions.

        Yields:
            Iterator[ChatSession]: The loaded chat sessions.
        