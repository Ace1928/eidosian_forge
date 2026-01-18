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
Lazy load the messages from the chat file and yield them
        in as chat sessions.

        Yields:
            ChatSession: The loaded chat session.
        