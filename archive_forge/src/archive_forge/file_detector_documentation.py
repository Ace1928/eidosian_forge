from abc import ABCMeta
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Optional
from selenium.types import AnyKey
from selenium.webdriver.common.utils import keys_to_typing
Detects files on the local disk.