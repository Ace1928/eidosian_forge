from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union, cast
from langchain_core.chat_sessions import ChatSession
from langchain_core.load.load import load
from langchain_community.chat_loaders.base import BaseChatLoader

        Lazy load the chat sessions from the specified LangSmith dataset.

        This method fetches the chat data from the dataset and
        converts each data point to chat sessions on-the-fly,
        yielding one session at a time.

        :return: Iterator of chat sessions containing messages.
        