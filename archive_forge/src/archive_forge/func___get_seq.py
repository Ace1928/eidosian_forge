from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def __get_seq(self):
    if not hasattr(self, '_seq'):
        self._seq = _retrieve_seq(self._adaptor, self._primary_id)
    return self._seq