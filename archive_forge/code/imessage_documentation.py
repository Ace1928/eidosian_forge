from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader

        Lazy load the chat sessions from the iMessage chat.db
        and yield them in the required format.

        Yields:
            ChatSession: Loaded chat session.
        