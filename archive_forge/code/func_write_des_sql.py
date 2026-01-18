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
def write_des_sql(self, handle):
    """Write DES data to SQL database."""
    cur = handle.cursor()
    cur.execute('DROP TABLE IF EXISTS des')
    cur.execute('CREATE TABLE des (sunid INT, type CHAR(2), sccs CHAR(10), description VARCHAR(255), PRIMARY KEY (sunid) )')
    for n in self._sunidDict.values():
        cur.execute('INSERT INTO des VALUES (%s,%s,%s,%s)', (n.sunid, n.type, n.sccs, n.description))