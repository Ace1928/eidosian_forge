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
def write_des(self, handle):
    """Build a DES SCOP parsable file from this object."""
    for n in sorted(self._sunidDict.values(), key=lambda x: x.sunid):
        if n != self.root:
            handle.write(str(n.toDesRecord()))