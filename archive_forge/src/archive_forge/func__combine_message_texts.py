from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _combine_message_texts(self, message_threads: Dict[int, List[int]], data: pd.DataFrame) -> str:
    """
        Combine the message texts for each parent message ID based             on the list of message threads.

        Args:
            message_threads (dict): A dictionary where the key is the parent message                 ID and the value is a list of message IDs in ascending order.
            data (pd.DataFrame): A DataFrame containing the conversation data:
                - message.sender_id
                - text
                - date
                - message.id
                - is_reply
                - reply_to_id

        Returns:
            str: A combined string of message texts sorted by date.
        """
    combined_text = ''
    for parent_id, message_ids in message_threads.items():
        message_texts = data[data['message.id'].isin(message_ids)].sort_values(by='date')['text'].tolist()
        message_texts = [str(elem) for elem in message_texts]
        combined_text += ' '.join(message_texts) + '.\n'
    return combined_text.strip()