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
def write_cla_sql(self, handle):
    """Write CLA data to SQL database."""
    cur = handle.cursor()
    cur.execute('DROP TABLE IF EXISTS cla')
    cur.execute('CREATE TABLE cla (sunid INT, sid CHAR(8), pdbid CHAR(4), residues VARCHAR(50), sccs CHAR(10), cl INT, cf INT, sf INT, fa INT, dm INT, sp INT, px INT, PRIMARY KEY (sunid), INDEX (SID) )')
    for n in self._sidDict.values():
        c = n.toClaRecord()
        cur.execute('INSERT INTO cla VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', (n.sunid, n.sid, c.residues.pdbid, c.residues, n.sccs, n.getAscendent('cl').sunid, n.getAscendent('cf').sunid, n.getAscendent('sf').sunid, n.getAscendent('fa').sunid, n.getAscendent('dm').sunid, n.getAscendent('sp').sunid, n.sunid))