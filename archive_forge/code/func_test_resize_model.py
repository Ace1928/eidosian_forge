from functools import partial
import pytest
from thinc.api import Linear, resizable
from thinc.layers.resizable import resize_linear_weighted, resize_model
def test_resize_model(model):
    """Test that resizing the model doesn't cause an exception."""
    resize_model(model, new_nO=10)
    resize_model(model, new_nO=11)
    model.set_dim('nO', 0, force=True)
    resize_model(model, new_nO=10)
    model.set_dim('nI', 10, force=True)
    model.set_dim('nO', 0, force=True)
    resize_model(model, new_nO=10)