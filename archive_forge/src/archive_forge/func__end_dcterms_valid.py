from ..datetimes import _parse_date
from ..util import FeedParserDict
def _end_dcterms_valid(self):
    for validity_detail in self.pop('validity').split(';'):
        if '=' in validity_detail:
            key, value = validity_detail.split('=', 1)
            if key == 'start':
                self._save('validity_start', value, overwrite=True)
                self._save('validity_start_parsed', _parse_date(value), overwrite=True)
            elif key == 'end':
                self._save('validity_end', value, overwrite=True)
                self._save('validity_end_parsed', _parse_date(value), overwrite=True)