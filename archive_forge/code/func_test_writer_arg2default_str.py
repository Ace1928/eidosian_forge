import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_writer_arg2default_str(self):
    self.writer_arg2default(0, path=None)
    self.writer_arg2default(0, path=self.name)
    assert self.read(self.name) == ''.join(self.text)