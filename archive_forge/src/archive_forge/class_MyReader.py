from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
class MyReader:

    def __init__(self, contents) -> None:
        self.read_count = 0
        self.stringio = StringIO(contents)

    def read(self, *args):
        self.read_count += 1
        return self.stringio.read(*args)

    def __iter__(self) -> Iterator:
        self.read_count += 1
        return iter(self.stringio)