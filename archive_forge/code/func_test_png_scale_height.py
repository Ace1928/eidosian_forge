import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('sizing_mode', ['scale_height', 'stretch_height'])
@pytest.mark.parametrize('embed', [False, True])
def test_png_scale_height(sizing_mode, embed, page):
    png = PNG(PNG_FILE, sizing_mode=sizing_mode, fixed_aspect=True, embed=embed)
    row = Row(png, height=500)
    bbox = get_bbox(page, row)
    assert int(bbox['width']) == 653
    assert bbox['height'] == 490