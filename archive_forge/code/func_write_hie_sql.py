import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def write_hie_sql(self, handle):
    """Write HIE data to SQL database."""
    cur = handle.cursor()
    cur.execute('DROP TABLE IF EXISTS hie')
    cur.execute('CREATE TABLE hie (parent INT, child INT, PRIMARY KEY (child), INDEX (parent) )')
    for p in self._sunidDict.values():
        for c in p.children:
            cur.execute(f'INSERT INTO hie VALUES ({p.sunid},{c.sunid})')