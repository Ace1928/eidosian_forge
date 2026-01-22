import os
import nbconvert
import nbformat
from holoviews.element.comparison import ComparisonTestCase
from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
def apply_preprocessors(preprocessors, nbname):
    notebooks_path = os.path.join(os.path.split(__file__)[0], 'notebooks')
    with open(os.path.join(notebooks_path, nbname)) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        exporter = nbconvert.PythonExporter()
        for preprocessor in preprocessors:
            exporter.register_preprocessor(preprocessor)
        source, meta = exporter.from_notebook_node(nb)
    return source