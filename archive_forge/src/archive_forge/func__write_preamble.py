import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_preamble(self, qresult):
    """Write the XML file preamble (PRIVATE)."""
    xml = self.xml
    for elem, attr in _WRITE_MAPS['preamble']:
        elem = 'BlastOutput_' + elem
        if elem == 'BlastOutput_param':
            xml.startParent(elem)
            self._write_param(qresult)
            xml.endParent()
            continue
        try:
            content = str(getattr(qresult, attr))
        except AttributeError:
            if elem not in _DTD_OPT:
                raise ValueError(f'Element {elem} (attribute {attr}) not found')
        else:
            if elem == 'BlastOutput_version':
                content = f'{qresult.program.upper()} {qresult.version}'
            elif qresult.blast_id:
                if elem == 'BlastOutput_query-ID':
                    content = qresult.blast_id
                elif elem == 'BlastOutput_query-def':
                    content = ' '.join([qresult.id, qresult.description]).strip()
            xml.simpleElement(elem, content)