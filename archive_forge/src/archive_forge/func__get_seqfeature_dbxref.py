from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_seqfeature_dbxref(self, seqfeature_id, dbxref_id, rank):
    """Get DB cross-reference, creating it if needed (PRIVATE).

        Check for a pre-existing seqfeature_dbxref entry with the passed
        seqfeature_id and dbxref_id.  If one does not exist, insert new
        data.
        """
    sql = 'SELECT seqfeature_id, dbxref_id FROM seqfeature_dbxref WHERE seqfeature_id = %s AND dbxref_id = %s'
    result = self.adaptor.execute_and_fetch_col0(sql, (seqfeature_id, dbxref_id))
    if result:
        return result
    return self._add_seqfeature_dbxref(seqfeature_id, dbxref_id, rank)