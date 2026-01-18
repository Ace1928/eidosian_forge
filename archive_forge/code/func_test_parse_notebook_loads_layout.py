from io import StringIO
import pytest
from panel.io.handlers import capture_code_cell, extract_code, parse_notebook
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import panel"""
@nbformat_available
def test_parse_notebook_loads_layout():
    cell = nbformat.v4.new_code_cell('1+1', metadata={'panel-layout': 'foo'})
    nb = nbformat.v4.new_notebook(cells=[cell])
    sio = StringIO(nbformat.v4.writes(nb))
    nb, code, layout = parse_notebook(sio)
    assert layout == {cell.id: 'foo'}
    assert code.startswith(f"_pn__state._cell_outputs['{cell.id}'].append((1+1))")