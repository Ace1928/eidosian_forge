import os
import nbconvert
import nbformat
from holoviews.element.comparison import ComparisonTestCase
from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
def test_opts_image_line_magic_svg(self):
    nbname = 'test_output_svg_line_magic.ipynb'
    expected = 'hv.util.output(" fig=\'svg\'")'
    source = apply_preprocessors([OutputMagicProcessor()], nbname)
    self.assertEqual(source.strip().endswith(expected), True)