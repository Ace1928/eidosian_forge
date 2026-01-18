import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_writer_arg1_str(self):
    self.writer_arg1(self.name)
    assert self.read(self.name) == ''.join(self.text)