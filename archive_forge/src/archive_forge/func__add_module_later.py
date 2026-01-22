import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence, Set, List
from .. import importcompletion
def _add_module_later(self, path: str) -> None:
    self.modules_to_add_later.append(path)