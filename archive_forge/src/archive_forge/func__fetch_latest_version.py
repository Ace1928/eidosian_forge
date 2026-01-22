import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _fetch_latest_version(self, base_id):
    """Get the latest version number for the given base_id (PRIVATE)."""
    cur = self.dbh.cursor()
    cur.execute('select VERSION from MATRIX where BASE_id = %s order by VERSION desc limit 1', (base_id,))
    row = cur.fetchone()
    latest = None
    if row:
        latest = row[0]
    else:
        warnings.warn(f"Failed to fetch latest version number for JASPAR motif with base ID '{base_id}'. No JASPAR motif with this base ID appears to exist in the database.", BiopythonWarning)
    return latest