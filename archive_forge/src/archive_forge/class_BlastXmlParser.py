import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastXmlParser:
    """Parser for the BLAST XML format."""

    def __init__(self, handle, use_raw_query_ids=False, use_raw_hit_ids=False):
        """Initialize the class."""
        self.xml_iter = iter(ElementTree.iterparse(handle, events=('start', 'end')))
        self._use_raw_query_ids = use_raw_query_ids
        self._use_raw_hit_ids = use_raw_hit_ids
        self._meta, self._fallback = self._parse_preamble()

    def __iter__(self):
        """Iterate over BlastXmlParser object yields query results."""
        yield from self._parse_qresult()

    def _parse_preamble(self):
        """Parse all tag data prior to the first query result (PRIVATE)."""
        meta = {}
        fallback = {}
        for event, elem in self.xml_iter:
            if event == 'end' and elem.tag in _ELEM_META:
                attr_name, caster = _ELEM_META[elem.tag]
                if caster is not str:
                    meta[attr_name] = caster(elem.text)
                else:
                    meta[attr_name] = elem.text
                elem.clear()
                continue
            elif event == 'end' and elem.tag in _ELEM_QRESULT_FALLBACK:
                attr_name, caster = _ELEM_QRESULT_FALLBACK[elem.tag]
                if caster is not str:
                    fallback[attr_name] = caster(elem.text)
                else:
                    fallback[attr_name] = elem.text
                elem.clear()
                continue
            if event == 'start' and elem.tag == 'Iteration':
                break
        if meta.get('version') is not None:
            meta['version'] = re.search(_RE_VERSION, meta['version']).group(0)
        return (meta, fallback)

    def _parse_qresult(self):
        """Parse query results (PRIVATE)."""
        for event, qresult_elem in self.xml_iter:
            if event == 'end' and qresult_elem.tag == 'Iteration':
                query_id = qresult_elem.findtext('Iteration_query-ID')
                if query_id is None:
                    query_id = self._fallback['id']
                query_desc = qresult_elem.findtext('Iteration_query-def')
                if query_desc is None:
                    query_desc = self._fallback['description']
                query_len = qresult_elem.findtext('Iteration_query-len')
                if query_len is None:
                    query_len = self._fallback['len']
                blast_query_id = query_id
                if not self._use_raw_query_ids and query_id.startswith(('Query_', 'lcl|')):
                    id_desc = query_desc.split(' ', 1)
                    query_id = id_desc[0]
                    try:
                        query_desc = id_desc[1]
                    except IndexError:
                        query_desc = ''
                hit_list, key_list = ([], [])
                for hit in self._parse_hit(qresult_elem.find('Iteration_hits'), query_id):
                    if hit:
                        if hit.id in key_list:
                            warnings.warn('Renaming hit ID %r to a BLAST-generated ID %r since the ID was already matched by your query %r. Your BLAST database may contain duplicate entries.' % (hit.id, hit.blast_id, query_id), BiopythonParserWarning)
                            hit.description = f'{hit.id} {hit.description}'
                            hit.id = hit.blast_id
                            for hsp in hit:
                                hsp.hit_id = hit.blast_id
                        else:
                            key_list.append(hit.id)
                        hit_list.append(hit)
                qresult = QueryResult(hit_list, query_id)
                qresult.description = query_desc
                qresult.seq_len = int(query_len)
                qresult.blast_id = blast_query_id
                for key, value in self._meta.items():
                    setattr(qresult, key, value)
                stat_iter_elem = qresult_elem.find('Iteration_stat')
                if stat_iter_elem is not None:
                    stat_elem = stat_iter_elem.find('Statistics')
                    for key, val_info in _ELEM_QRESULT_OPT.items():
                        value = stat_elem.findtext(key)
                        if value is not None:
                            caster = val_info[1]
                            if value is not None and caster is not str:
                                value = caster(value)
                            setattr(qresult, val_info[0], value)
                qresult_elem.clear()
                yield qresult

    def _parse_hit(self, root_hit_elem, query_id):
        """Yield a generator object that transforms Iteration_hits XML elements into Hit objects (PRIVATE).

        :param root_hit_elem: root element of the Iteration_hits tag.
        :type root_hit_elem: XML element tag
        :param query_id: QueryResult ID of this Hit
        :type query_id: string

        """
        if root_hit_elem is None:
            root_hit_elem = []
        for hit_elem in root_hit_elem:
            raw_hit_id = hit_elem.findtext('Hit_id')
            raw_hit_desc = hit_elem.findtext('Hit_def')
            if not self._use_raw_hit_ids:
                ids, descs, blast_hit_id = _extract_ids_and_descs(raw_hit_id, raw_hit_desc)
            else:
                ids, descs, blast_hit_id = ([raw_hit_id], [raw_hit_desc], raw_hit_id)
            hit_id, alt_hit_ids = (ids[0], ids[1:])
            hit_desc, alt_hit_descs = (descs[0], descs[1:])
            hsps = list(self._parse_hsp(hit_elem.find('Hit_hsps'), query_id, hit_id))
            hit = Hit(hsps)
            hit.description = hit_desc
            hit._id_alt = alt_hit_ids
            hit._description_alt = alt_hit_descs
            hit.blast_id = blast_hit_id
            for key, val_info in _ELEM_HIT.items():
                value = hit_elem.findtext(key)
                if value is not None:
                    caster = val_info[1]
                    if value is not None and caster is not str:
                        value = caster(value)
                    setattr(hit, val_info[0], value)
            hit_elem.clear()
            yield hit

    def _parse_hsp(self, root_hsp_frag_elem, query_id, hit_id):
        """Yield a generator object that transforms Hit_hsps XML elements into HSP objects (PRIVATE).

        :param root_hsp_frag_elem: the ``Hit_hsps`` tag
        :type root_hsp_frag_elem: XML element tag
        :param query_id: query ID
        :type query_id: string
        :param hit_id: hit ID
        :type hit_id: string

        """
        if root_hsp_frag_elem is None:
            root_hsp_frag_elem = []
        for hsp_frag_elem in root_hsp_frag_elem:
            coords = {}
            frag = HSPFragment(hit_id, query_id)
            for key, val_info in _ELEM_FRAG.items():
                value = hsp_frag_elem.findtext(key)
                caster = val_info[1]
                if value is not None:
                    if key.endswith(('-from', '-to')):
                        coords[val_info[0]] = caster(value)
                        continue
                    elif caster is not str:
                        value = caster(value)
                    setattr(frag, val_info[0], value)
            frag.aln_annotation['similarity'] = hsp_frag_elem.findtext('Hsp_midline')
            for coord_type in ('query', 'hit', 'pattern'):
                start_type = coord_type + '_start'
                end_type = coord_type + '_end'
                try:
                    start = coords[start_type]
                    end = coords[end_type]
                except KeyError:
                    continue
                else:
                    setattr(frag, start_type, min(start, end) - 1)
                    setattr(frag, end_type, max(start, end))
            prog = self._meta.get('program')
            if prog == 'blastn':
                frag.molecule_type = 'DNA'
            elif prog in ['blastp', 'blastx', 'tblastn', 'tblastx']:
                frag.molecule_type = 'protein'
            hsp = HSP([frag])
            for key, val_info in _ELEM_HSP.items():
                value = hsp_frag_elem.findtext(key)
                caster = val_info[1]
                if value is not None:
                    if caster is not str:
                        value = caster(value)
                    setattr(hsp, val_info[0], value)
            hsp_frag_elem.clear()
            yield hsp