from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _update_left_right_taxon_values(self, left_value):
    """Update the left and right taxon values in the table (PRIVATE)."""
    if not left_value:
        return
    rows = self.adaptor.execute_and_fetchall('SELECT left_value, right_value, taxon_id FROM taxon WHERE right_value >= %s or left_value > %s', (left_value, left_value))
    right_rows = []
    left_rows = []
    for row in rows:
        new_right = row[1]
        new_left = row[0]
        if new_right >= left_value:
            new_right += 2
        if new_left > left_value:
            new_left += 2
        right_rows.append((new_right, row[2]))
        left_rows.append((new_left, row[2]))
    right_rows = sorted(right_rows, key=lambda x: x[0], reverse=True)
    left_rows = sorted(left_rows, key=lambda x: x[0], reverse=True)
    self.adaptor.executemany('UPDATE taxon SET left_value = %s WHERE taxon_id = %s', left_rows)
    self.adaptor.executemany('UPDATE taxon SET right_value = %s WHERE taxon_id = %s', right_rows)