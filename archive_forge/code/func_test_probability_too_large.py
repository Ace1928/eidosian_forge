import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_probability_too_large(self):
    with pytest.raises(NetworkXError):
        duplication_divergence_graph(3, 2)