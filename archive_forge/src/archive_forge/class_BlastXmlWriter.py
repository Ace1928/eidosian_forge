import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastXmlWriter:
    """Stream-based BLAST+ XML Writer."""

    def __init__(self, handle, use_raw_query_ids=True, use_raw_hit_ids=True):
        """Initialize the class."""
        self.xml = _BlastXmlGenerator(handle, 'utf-8')
        self._use_raw_query_ids = use_raw_query_ids
        self._use_raw_hit_ids = use_raw_hit_ids

    def write_file(self, qresults):
        """Write the XML contents to the output handle."""
        xml = self.xml
        self.qresult_counter, self.hit_counter, self.hsp_counter, self.frag_counter = (0, 0, 0, 0)
        first_qresult = next(qresults)
        xml.startDocument()
        xml.startParent('BlastOutput')
        self._write_preamble(first_qresult)
        xml.startParent('BlastOutput_iterations')
        self._write_qresults(chain([first_qresult], qresults))
        xml.endParents(2)
        xml.endDocument()
        return (self.qresult_counter, self.hit_counter, self.hsp_counter, self.frag_counter)

    def _write_elem_block(self, block_name, map_name, obj, opt_dict=None):
        """Write sibling XML elements (PRIVATE).

        :param block_name: common element name prefix
        :type block_name: string
        :param map_name: name of mapping between element and attribute names
        :type map_name: string
        :param obj: object whose attribute value will be used
        :type obj: object
        :param opt_dict: custom element-attribute mapping
        :type opt_dict: dictionary {string: string}

        """
        if opt_dict is None:
            opt_dict = {}
        for elem, attr in _WRITE_MAPS[map_name]:
            elem = block_name + elem
            try:
                content = str(getattr(obj, attr))
            except AttributeError:
                if elem not in _DTD_OPT:
                    raise ValueError(f'Element {elem!r} (attribute {attr!r}) not found')
            else:
                if elem in opt_dict:
                    content = opt_dict[elem]
                self.xml.simpleElement(elem, content)

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

    def _write_param(self, qresult):
        """Write the parameter block of the preamble (PRIVATE)."""
        xml = self.xml
        xml.startParent('Parameters')
        self._write_elem_block('Parameters_', 'param', qresult)
        xml.endParent()

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

    def _write_hits(self, hits):
        """Write Hit objects (PRIVATE)."""
        xml = self.xml
        for num, hit in enumerate(hits):
            xml.startParent('Hit')
            xml.simpleElement('Hit_num', str(num + 1))
            opt_dict = {}
            if self._use_raw_hit_ids:
                hit_id = hit.blast_id
                hit_desc = ' >'.join([f'{x} {y}' for x, y in zip(hit.id_all, hit.description_all)])
            else:
                hit_id = hit.id
                hit_desc = hit.description + ' >'.join([f'{x} {y}' for x, y in zip(hit.id_all[1:], hit.description_all[1:])])
            opt_dict = {'Hit_id': hit_id, 'Hit_def': hit_desc}
            self._write_elem_block('Hit_', 'hit', hit, opt_dict)
            xml.startParent('Hit_hsps')
            self._write_hsps(hit.hsps)
            self.hit_counter += 1
            xml.endParents(2)

    def _write_hsps(self, hsps):
        """Write HSP objects (PRIVATE)."""
        xml = self.xml
        for num, hsp in enumerate(hsps):
            xml.startParent('Hsp')
            xml.simpleElement('Hsp_num', str(num + 1))
            for elem, attr in _WRITE_MAPS['hsp']:
                elem = 'Hsp_' + elem
                try:
                    content = self._adjust_output(hsp, elem, attr)
                except AttributeError:
                    if elem not in _DTD_OPT:
                        raise ValueError(f'Element {elem} (attribute {attr}) not found')
                else:
                    xml.simpleElement(elem, str(content))
            self.hsp_counter += 1
            self.frag_counter += len(hsp.fragments)
            xml.endParent()

    def _adjust_output(self, hsp, elem, attr):
        """Adjust output to mimic native BLAST+ XML as much as possible (PRIVATE)."""
        if attr in ('query_start', 'query_end', 'hit_start', 'hit_end', 'pattern_start', 'pattern_end'):
            content = getattr(hsp, attr) + 1
            if '_start' in attr:
                content = getattr(hsp, attr) + 1
            else:
                content = getattr(hsp, attr)
            if hsp.query_frame != 0 and hsp.hit_frame < 0:
                if attr == 'hit_start':
                    content = getattr(hsp, 'hit_end')
                elif attr == 'hit_end':
                    content = getattr(hsp, 'hit_start') + 1
        elif elem in ('Hsp_hseq', 'Hsp_qseq'):
            content = str(getattr(hsp, attr).seq)
        elif elem == 'Hsp_midline':
            content = hsp.aln_annotation['similarity']
        elif elem in ('Hsp_evalue', 'Hsp_bit-score'):
            content = '%.*g' % (6, getattr(hsp, attr))
        else:
            content = getattr(hsp, attr)
        return content