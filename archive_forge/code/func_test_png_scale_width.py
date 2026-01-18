import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('sizing_mode', ['scale_width', 'stretch_width'])
@pytest.mark.parametrize('embed', [False, True])
def test_png_scale_width(sizing_mode, embed, page):
    png = PNG(PNG_FILE, sizing_mode=sizing_mode, fixed_aspect=True, embed=embed)
    row = Row(png, width=800)
    bbox = get_bbox(page, row)
    assert bbox['width'] == 780
    assert bbox['height'] == 585