import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_writer_arg2default_fobj_path_none(self):
    self.writer_arg2default(0, path=None)