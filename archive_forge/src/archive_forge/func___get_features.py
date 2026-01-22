from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def __get_features(self):
    if not hasattr(self, '_features'):
        self._features = _retrieve_features(self._adaptor, self._primary_id)
    return self._features