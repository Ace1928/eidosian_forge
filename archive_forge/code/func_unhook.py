import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def unhook(self, name):
    handle = self._hook_handles.pop(name)
    handle.remove()