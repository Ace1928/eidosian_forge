from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
def _huggingface_tokenizer_length(text: str) -> int:
    return len(tokenizer.encode(text))