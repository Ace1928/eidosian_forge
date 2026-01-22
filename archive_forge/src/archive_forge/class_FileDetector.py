from abc import ABCMeta
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Optional
from selenium.types import AnyKey
from selenium.webdriver.common.utils import keys_to_typing
class FileDetector(metaclass=ABCMeta):
    """Used for identifying whether a sequence of chars represents the path to
    a file."""

    @abstractmethod
    def is_local_file(self, *keys: AnyKey) -> Optional[str]:
        raise NotImplementedError