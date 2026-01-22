from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_ontology_id(self, name, definition=None):
    """Return identifier for the named ontology (PRIVATE).

        This looks through the onotology table for a the given entry name.
        If it is not found, a row is added for this ontology (using the
        definition if supplied).  In either case, the id corresponding to
        the provided name is returned, so that you can reference it in
        another table.
        """
    oids = self.adaptor.execute_and_fetch_col0('SELECT ontology_id FROM ontology WHERE name = %s', (name,))
    if oids:
        return oids[0]
    self.adaptor.execute('INSERT INTO ontology(name, definition) VALUES (%s, %s)', (name, definition))
    return self.adaptor.last_id('ontology')