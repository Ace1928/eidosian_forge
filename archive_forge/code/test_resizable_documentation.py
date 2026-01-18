from functools import partial
import pytest
from thinc.api import Linear, resizable
from thinc.layers.resizable import resize_linear_weighted, resize_model
Test that resizing the model doesn't cause an exception.