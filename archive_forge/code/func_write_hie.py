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
def write_hie(self, handle):
    """Build an HIE SCOP parsable file from this object."""
    for n in sorted(self._sunidDict.values(), key=lambda x: x.sunid):
        handle.write(str(n.toHieRecord()))