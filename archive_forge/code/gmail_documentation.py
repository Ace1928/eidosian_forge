import base64
import re
from typing import Any, Iterator
from langchain_core._api.deprecation import deprecated
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
Load data from `GMail`.

    There are many ways you could want to load data from GMail.
    This loader is currently fairly opinionated in how to do so.
    The way it does it is it first looks for all messages that you have sent.
    It then looks for messages where you are responding to a previous email.
    It then fetches that previous email, and creates a training example
    of that email, followed by your email.

    Note that there are clear limitations here. For example,
    all examples created are only looking at the previous email for context.

    To use:

    - Set up a Google Developer Account:
        Go to the Google Developer Console, create a project,
        and enable the Gmail API for that project.
        This will give you a credentials.json file that you'll need later.
    