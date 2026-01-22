from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_seqfeature_qualifiers(self, qualifiers, seqfeature_id):
    """Insert feature's (key, value) pair qualifiers (PRIVATE).

        Qualifiers should be a dictionary of the form::

            {key : [value1, value2]}

        """
    tag_ontology_id = self._get_ontology_id('Annotation Tags')
    for qualifier_key in qualifiers:
        if qualifier_key != 'db_xref':
            qualifier_key_id = self._get_term_id(qualifier_key, ontology_id=tag_ontology_id)
            entries = qualifiers[qualifier_key]
            if not isinstance(entries, list):
                entries = [entries]
            for qual_value_rank in range(len(entries)):
                qualifier_value = entries[qual_value_rank]
                sql = 'INSERT INTO seqfeature_qualifier_value  (seqfeature_id, term_id, "rank", value) VALUES (%s, %s, %s, %s)'
                self.adaptor.execute(sql, (seqfeature_id, qualifier_key_id, qual_value_rank + 1, qualifier_value))
        else:
            self._load_seqfeature_dbxref(qualifiers[qualifier_key], seqfeature_id)