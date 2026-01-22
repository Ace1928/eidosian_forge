from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _get_board(self) -> Board:
    board = next((b for b in self.client.list_boards() if b.name == self.board_name), None)
    if not board:
        raise ValueError(f'Board `{self.board_name}` not found.')
    return board