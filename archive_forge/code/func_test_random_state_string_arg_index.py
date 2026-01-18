import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_random_state_string_arg_index():
    with pytest.raises(nx.NetworkXError):

        @np_random_state('a')
        def make_random_state(rs):
            pass
        rstate = make_random_state(1)