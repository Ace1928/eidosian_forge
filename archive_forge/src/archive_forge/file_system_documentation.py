import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.stores import ByteStore
from langchain.storage.exceptions import InvalidKeyException
Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        