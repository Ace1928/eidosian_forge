import errno
import subprocess
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
import requests
import unicodedata
from .config import getpreferredencoding
from .translations import _
from ._typing_compat import Protocol
class Paster(Protocol):

    def paste(self, s: str) -> Tuple[str, Optional[str]]:
        ...