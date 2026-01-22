from contextlib import contextmanager
from jedi import debug
from jedi.inference.base_value import NO_VALUES
class RecursionDetector:

    def __init__(self):
        self.pushed_nodes = []