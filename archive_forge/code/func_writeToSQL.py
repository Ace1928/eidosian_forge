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
def writeToSQL(self, db_handle):
    """Write the ASTRAL database to a MYSQL database."""
    cur = db_handle.cursor()
    cur.execute('DROP TABLE IF EXISTS astral')
    cur.execute('CREATE TABLE astral (sid CHAR(8), seq TEXT, PRIMARY KEY (sid))')
    for dom in self.fasta_dict:
        cur.execute('INSERT INTO astral (sid,seq) values (%s,%s)', (dom, self.fasta_dict[dom].seq))
    for i in astralBibIds:
        cur.execute('ALTER TABLE astral ADD (id' + str(i) + ' TINYINT)')
        for d in self.domainsClusteredById(i):
            cur.execute('UPDATE astral SET id' + str(i) + '=1  WHERE sid=%s', d.sid)
    for ev in astralEvs:
        cur.execute('ALTER TABLE astral ADD (' + astralEv_to_sql[ev] + ' TINYINT)')
        for d in self.domainsClusteredByEv(ev):
            cur.execute('UPDATE astral SET ' + astralEv_to_sql[ev] + '=1  WHERE sid=%s', d.sid)