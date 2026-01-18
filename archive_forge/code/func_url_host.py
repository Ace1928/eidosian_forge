from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def url_host(self) -> str:
    return f'{PREFIX_HTTPS}{self.host}' if self.host else ''