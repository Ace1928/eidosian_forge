import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, lists
from numpy.testing import assert_allclose
from thinc.layers import Embed
from thinc.layers.uniqued import uniqued
def test_uniqued_calls_init():
    calls = []
    embed = Embed(5, 5, column=0)
    embed.init = lambda *args, **kwargs: calls.append(True)
    embed.initialize()
    assert calls == [True]
    uembed = uniqued(embed)
    uembed.initialize()
    assert calls == [True, True]