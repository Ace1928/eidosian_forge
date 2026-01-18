import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_tryfinally_generator(self):
    container = []

    def singleton(x):
        return (x,)
    with pytest.raises(nx.NetworkXError):

        @argmap(singleton, 0, 1, 2, try_finally=True)
        def foo(x, y, z):
            yield from (x, y, z)

    @argmap(singleton, 0, 1, 2)
    def foo(x, y, z):
        return x + y + z
    q = foo('a', 'b', 'c')
    assert q == ('a', 'b', 'c')