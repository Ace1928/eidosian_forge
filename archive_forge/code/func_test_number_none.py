import pytest
from panel.widgets.indicators import (
def test_number_none(document, comm):
    number = Number(value=None, name='Value')
    model = number.get_root(document, comm)
    assert model.text.endswith('&lt;div style=&quot;font-size: 54pt; color: black&quot;&gt;-&lt;/div&gt;')
    number.nan_format = 'nan'
    assert model.text.endswith('&lt;div style=&quot;font-size: 54pt; color: black&quot;&gt;nan&lt;/div&gt;')