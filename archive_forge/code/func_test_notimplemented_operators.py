import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_notimplemented_operators(self):
    nl = rlc.OrdDict()
    nl2 = rlc.OrdDict()
    assert nl == nl
    assert nl != nl2
    with pytest.raises(TypeError):
        nl > nl2
    with pytest.raises(NotImplementedError):
        reversed(nl)
    with pytest.raises(NotImplementedError):
        nl.sort()