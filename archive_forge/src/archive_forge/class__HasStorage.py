import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
class _HasStorage:

    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage