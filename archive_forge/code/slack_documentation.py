import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader

        Lazy load the chat sessions from the Slack dump file and yield them
        in the required format.

        :return: Iterator of chat sessions containing messages.
        