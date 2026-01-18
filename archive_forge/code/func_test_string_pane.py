import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_string_pane(document, comm):
    pane = Str('<h1>Test</h1>')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;pre&gt;&lt;h1&gt;Test&lt;/h1&gt;&lt;/pre&gt;'
    pane.object = '<h2>Test</h2>'
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;pre&gt;&lt;h2&gt;Test&lt;/h2&gt;&lt;/pre&gt;'
    pane._cleanup(model)
    assert pane._models == {}