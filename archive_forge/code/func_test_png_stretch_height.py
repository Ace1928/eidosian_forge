import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('embed', [False, True])
def test_png_stretch_height(embed, page):
    png = PNG(PNG_FILE, sizing_mode='stretch_height', fixed_aspect=False, width=500, embed=embed)
    row = Row(png, height=500)
    bbox = get_bbox(page, row)
    assert bbox['width'] == 500
    assert bbox['height'] == 490