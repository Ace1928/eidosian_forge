from typing import Callable, Optional
from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore
from langchain.storage.encoder_backed import EncoderBackedStore
def _load_document_from_bytes(serialized: bytes) -> Document:
    """Return a document from a bytes representation."""
    obj = loads(serialized.decode('utf-8'))
    if not isinstance(obj, Document):
        raise TypeError(f'Expected a Document instance. Got {type(obj)}')
    return obj