import re
from ._base import _BaseExonerateParser, _BaseExonerateIndexer, _STRAND_MAP
from typing import Type
class ExonerateVulgarParser(_BaseExonerateParser):
    """Parser for Exonerate vulgar strings."""
    _ALN_MARK = 'vulgar'

    def parse_alignment_block(self, header):
        """Parse alignment block for vulgar format, return query results, hits, hsps."""
        qresult = header['qresult']
        hit = header['hit']
        hsp = header['hsp']
        self.read_until(lambda line: line.startswith('vulgar'))
        vulgars = re.search(_RE_VULGAR, self.line)
        if self.has_c4_alignment:
            assert qresult['id'] == vulgars.group(1)
            assert hsp['query_start'] == vulgars.group(2)
            assert hsp['query_end'] == vulgars.group(3)
            assert hsp['query_strand'] == vulgars.group(4)
            assert hit['id'] == vulgars.group(5)
            assert hsp['hit_start'] == vulgars.group(6)
            assert hsp['hit_end'] == vulgars.group(7)
            assert hsp['hit_strand'] == vulgars.group(8)
            assert hsp['score'] == vulgars.group(9)
        else:
            qresult['id'] = vulgars.group(1)
            hsp['query_start'] = vulgars.group(2)
            hsp['query_end'] = vulgars.group(3)
            hsp['query_strand'] = vulgars.group(4)
            hit['id'] = vulgars.group(5)
            hsp['hit_start'] = vulgars.group(6)
            hsp['hit_end'] = vulgars.group(7)
            hsp['hit_strand'] = vulgars.group(8)
            hsp['score'] = vulgars.group(9)
        hsp['hit_strand'] = _STRAND_MAP[hsp['hit_strand']]
        hsp['query_strand'] = _STRAND_MAP[hsp['query_strand']]
        hsp['query_start'] = int(hsp['query_start'])
        hsp['query_end'] = int(hsp['query_end'])
        hsp['hit_start'] = int(hsp['hit_start'])
        hsp['hit_end'] = int(hsp['hit_end'])
        hsp['score'] = int(hsp['score'])
        hsp['vulgar_comp'] = vulgars.group(10).rstrip()
        hsp = parse_vulgar_comp(hsp, hsp['vulgar_comp'])
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp}