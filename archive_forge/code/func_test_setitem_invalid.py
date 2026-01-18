import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_setitem_invalid():
    env = rinterface.baseenv['new.env']()
    with pytest.raises(TypeError):
        env[None] = 0
    with pytest.raises(ValueError):
        env[''] = 0