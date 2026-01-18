import pytest
from numpy.testing import assert_allclose
from panel.layout import Row
from panel.pane import PDF, PNG, SVG
from panel.tests.util import serve_component, wait_for_server
@pytest.mark.parametrize('embed', [False, True])
def test_png_native_size(embed, page):
    png = PNG(PNG_FILE, embed=embed)
    bbox = get_bbox(page, png)
    assert bbox['width'] == 800
    assert bbox['height'] == 600