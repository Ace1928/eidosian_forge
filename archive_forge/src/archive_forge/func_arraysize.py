import sqlite3
from typing import (
@arraysize.setter
def arraysize(self, value: int) -> None:
    self._cursor.arraysize = value