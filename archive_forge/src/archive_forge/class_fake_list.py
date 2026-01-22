import pytest
from jeepney.low_level import *
class fake_list(list):

    def __init__(self, n):
        super().__init__()
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(range(self._n))