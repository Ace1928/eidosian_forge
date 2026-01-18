import os
import nbconvert
import nbformat
from holoviews.element.comparison import ComparisonTestCase
from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
def test_opts_image_line_magic(self):
    nbname = 'test_opts_image_line_magic.ipynb'
    expected = 'hv.util.opts(" Image [xaxis=None] (cmap=\'viridis\')")'
    source = apply_preprocessors([OptsMagicProcessor()], nbname)
    self.assertEqual(source.strip().endswith(expected), True)