import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('embed', [False, True])
def test_svg_stretch_width(embed, page):
    svg = SVG(SVG_FILE, sizing_mode='stretch_width', fixed_aspect=False, embed=embed, height=500)
    row = Row(svg, width=1000)
    bbox = get_bbox(page, row)
    assert bbox['width'] == 980
    assert bbox['height'] == 500