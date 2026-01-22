import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _is_latest_version(self, int_id):
    """Check if the internal ID represents the latest JASPAR matrix (PRIVATE).

        Does this internal ID represent the latest version of the JASPAR
        matrix (collapse on base ids)
        """
    cur = self.dbh.cursor()
    cur.execute('select count(*) from MATRIX where BASE_ID = (select BASE_ID from MATRIX where ID = %s) and VERSION > (select VERSION from MATRIX where ID = %s)', (int_id, int_id))
    row = cur.fetchone()
    count = row[0]
    if count == 0:
        return True
    return False