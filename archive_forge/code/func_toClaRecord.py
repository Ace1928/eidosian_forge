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
def toClaRecord(self):
    """Return a Cla.Record."""
    rec = Cla.Record()
    rec.sid = self.sid
    rec.residues = self.residues
    rec.sccs = self.sccs
    rec.sunid = self.sunid
    n = self
    while n.sunid != 0:
        rec.hierarchy[n.type] = str(n.sunid)
        n = n.getParent()
    return rec