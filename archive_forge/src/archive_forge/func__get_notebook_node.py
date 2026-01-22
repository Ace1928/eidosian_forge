import os
import sys
import time
import traceback
import param
from IPython.display import Javascript, display
from nbconvert import HTMLExporter, NotebookExporter
from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbformat import reader
from ..core.io import FileArchive, Pickler
from ..plotting.renderer import HTML_TAGS, MIME_TYPES
from .preprocessors import Substitute
def _get_notebook_node(self):
    """Load captured notebook node"""
    size = len(self._notebook_data)
    if size == 0:
        raise Exception('Captured buffer size for notebook node is zero.')
    node = reader.reads(self._notebook_data)
    self.nbversion = reader.get_version(node)
    return node