import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@property
def total_pages(self) -> Optional[int]:
    """The total number of pages."""
    if self._total_pages is not None:
        return int(self._total_pages)
    return None