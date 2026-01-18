import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@not_implemented_for('undirected')
def test_u(G):
    pass