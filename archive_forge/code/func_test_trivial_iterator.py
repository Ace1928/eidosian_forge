import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_trivial_iterator(self):

    def do_not_call(x):
        raise ArgmapError('do not call this function')

    @argmap(do_not_call)
    def trivial_argmap():
        yield from (1, 2, 3)
    assert tuple(trivial_argmap()) == (1, 2, 3)