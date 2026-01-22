import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_qresults(self, qresults):
    """Write QueryResult objects into iteration elements (PRIVATE)."""
    xml = self.xml
    for num, qresult in enumerate(qresults):
        xml.startParent('Iteration')
        xml.simpleElement('Iteration_iter-num', str(num + 1))
        opt_dict = {}
        if self._use_raw_query_ids:
            query_id = qresult.blast_id
            query_desc = qresult.id + ' ' + qresult.description
        else:
            query_id = qresult.id
            query_desc = qresult.description
        opt_dict = {'Iteration_query-ID': query_id, 'Iteration_query-def': query_desc}
        self._write_elem_block('Iteration_', 'qresult', qresult, opt_dict)
        if qresult:
            xml.startParent('Iteration_hits')
            self._write_hits(qresult.hits)
            xml.endParent()
        else:
            xml.simpleElement('Iteration_hits', '')
        xml.startParents('Iteration_stat', 'Statistics')
        self._write_elem_block('Statistics_', 'stat', qresult)
        xml.endParents(2)
        if not qresult:
            xml.simpleElement('Iteration_message', 'No hits found')
        self.qresult_counter += 1
        xml.endParent()