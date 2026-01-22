import datetime
import math
import typing as t
from wandb.util import (
def _set_serialization_path(self, path: str, key: str) -> None:
    self._serialization_path = {'path': path, 'key': key}