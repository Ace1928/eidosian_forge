from __future__ import annotations
import hashlib
import json
import uuid
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, cast
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate
from langchain.storage.encoder_backed import EncoderBackedStore
def _key_encoder(key: str, namespace: str) -> str:
    """Encode a key."""
    return namespace + str(_hash_string_to_uuid(key))