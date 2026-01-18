import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_writer_arg4default_fobj(self):
    self.writer_arg4default(0, 1, dog='dog', other='other')
    self.writer_arg4default(0, 1, dog='dog', other='other', path=self.name)
    assert self.read(self.name) == ''.join(self.text)