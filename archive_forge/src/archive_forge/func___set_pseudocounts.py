from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
def __set_pseudocounts(self, value):
    self._pseudocounts = {}
    if isinstance(value, dict):
        self._pseudocounts = {letter: value[letter] for letter in self.alphabet}
    else:
        if value is None:
            value = 0.0
        self._pseudocounts = dict.fromkeys(self.alphabet, value)