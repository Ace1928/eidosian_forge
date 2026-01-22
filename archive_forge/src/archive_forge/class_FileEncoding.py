import concurrent.futures
from pathlib import Path
from typing import List, NamedTuple, Optional, Union, cast
class FileEncoding(NamedTuple):
    """File encoding as the NamedTuple."""
    encoding: Optional[str]
    'The encoding of the file.'
    confidence: float
    'The confidence of the encoding.'
    language: Optional[str]
    'The language of the file.'