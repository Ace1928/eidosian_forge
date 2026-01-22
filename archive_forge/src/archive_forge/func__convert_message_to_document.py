import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _convert_message_to_document(self, message: dict, channel_name: str) -> Document:
    """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
    text = message.get('text', '')
    metadata = self._get_message_metadata(message, channel_name)
    return Document(page_content=text, metadata=metadata)