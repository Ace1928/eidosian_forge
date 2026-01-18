import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        