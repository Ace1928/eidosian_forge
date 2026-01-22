from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _card_to_doc(self, card: Card, list_dict: dict) -> Document:
    from bs4 import BeautifulSoup
    text_content = ''
    if self.include_card_name:
        text_content = card.name + '\n'
    if card.description.strip():
        text_content += BeautifulSoup(card.description, 'lxml').get_text()
    if self.include_checklist:
        for checklist in card.checklists:
            if checklist.items:
                items = [f'{item['name']}:{item['state']}' for item in checklist.items]
                text_content += f'\n{checklist.name}\n' + '\n'.join(items)
    if self.include_comments:
        comments = [BeautifulSoup(comment['data']['text'], 'lxml').get_text() for comment in card.comments]
        text_content += 'Comments:' + '\n'.join(comments)
    metadata = {'title': card.name, 'id': card.id, 'url': card.url}
    if 'labels' in self.extra_metadata:
        metadata['labels'] = [label.name for label in card.labels]
    if 'list' in self.extra_metadata:
        if card.list_id in list_dict:
            metadata['list'] = list_dict[card.list_id]
    if 'closed' in self.extra_metadata:
        metadata['closed'] = card.closed
    if 'due_date' in self.extra_metadata:
        metadata['due_date'] = card.due_date
    return Document(page_content=text_content, metadata=metadata)