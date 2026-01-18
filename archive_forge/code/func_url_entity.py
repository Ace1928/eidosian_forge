from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def url_entity(self) -> str:
    assert self.entity
    return f'{self.url_host()}/{self.entity}'