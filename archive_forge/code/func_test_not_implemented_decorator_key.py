import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_not_implemented_decorator_key():
    with pytest.raises(KeyError):

        @not_implemented_for('foo')
        def test1(G):
            pass
        test1(nx.Graph())