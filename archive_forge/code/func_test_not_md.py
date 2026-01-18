import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@not_implemented_for('directed', 'multigraph')
def test_not_md(G):
    pass