import re
from xml.etree import ElementTree
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_hit(self, root_hit_elem, query_id, query_seq=None):
    """Parse hit (PRIVATE)."""
    if root_hit_elem is None:
        root_hit_elem = []
    for hit_elem in root_hit_elem:
        hit_type = re.sub('%s(\\w+)-match' % self.NS, '\\1', hit_elem.find('.').tag)
        signature = hit_elem.find(self.NS + 'signature')
        hit_id = signature.attrib['ac']
        xrefs = self._parse_xrefs(signature.find(self.NS + 'entry'))
        hsps = list(self._parse_hsp(hit_elem.find(self.NS + 'locations'), query_id, hit_id, query_seq))
        hit = Hit(hsps, hit_id)
        setattr(hit, 'dbxrefs', xrefs)
        for key, (attr, caster) in _ELEM_HIT.items():
            value = signature.attrib.get(key)
            if value is not None:
                setattr(hit, attr, caster(value))
        hit.attributes['Hit type'] = hit_type
        signature_lib = signature.find(self.NS + 'signature-library-release')
        hit.attributes['Target'] = str(signature_lib.attrib.get('library'))
        hit.attributes['Target version'] = str(signature_lib.attrib.get('version'))
        yield hit