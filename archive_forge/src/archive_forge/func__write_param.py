import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_param(self, qresult):
    """Write the parameter block of the preamble (PRIVATE)."""
    xml = self.xml
    xml.startParent('Parameters')
    self._write_elem_block('Parameters_', 'param', qresult)
    xml.endParent()