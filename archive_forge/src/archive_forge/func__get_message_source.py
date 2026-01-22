import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
    """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
    if self.workspace_url:
        channel_id = self.channel_id_map.get(channel_name, '')
        return f'{self.workspace_url}/archives/{channel_id}' + f'/p{timestamp.replace('.', '')}'
    else:
        return f'{channel_name} - {user} - {timestamp}'