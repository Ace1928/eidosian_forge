import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class MySubclassWithMetadata(DataFrame):
    _metadata = ['my_metadata']

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        my_metadata = kwargs.pop('my_metadata', None)
        if args and isinstance(args[0], MySubclassWithMetadata):
            my_metadata = args[0].my_metadata
        self.my_metadata = my_metadata

    @property
    def _constructor(self):
        return MySubclassWithMetadata