from io import StringIO
import pytest
from panel.io.handlers import capture_code_cell, extract_code, parse_notebook
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import panel"""
@nbformat_available
def test_parse_notebook_markdown_escaped():
    cell = nbformat.v4.new_markdown_cell('This is a test of markdown terminated by a quote"')
    nb = nbformat.v4.new_notebook(cells=[cell])
    sio = StringIO(nbformat.v4.writes(nb))
    nb, code, layout = parse_notebook(sio)
    assert code == f'''_pn__state._cell_outputs['{cell.id}'].append("""This is a test of markdown terminated by a quote\\"""")'''