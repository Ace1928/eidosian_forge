from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(self, encoding_name: str='gpt2', model_name: Optional[str]=None, allowed_special: Union[Literal['all'], AbstractSet[str]]=set(), disallowed_special: Union[Literal['all'], Collection[str]]='all', **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError('Could not import tiktoken python package. This is needed in order to for TokenTextSplitter. Please install it with `pip install tiktoken`.')
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:

        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(_text, allowed_special=self._allowed_special, disallowed_special=self._disallowed_special)
        tokenizer = Tokenizer(chunk_overlap=self._chunk_overlap, tokens_per_chunk=self._chunk_size, decode=self._tokenizer.decode, encode=_encode)
        return split_text_on_tokens(text=text, tokenizer=tokenizer)