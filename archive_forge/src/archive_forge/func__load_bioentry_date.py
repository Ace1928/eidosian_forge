from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_bioentry_date(self, record, bioentry_id):
    """Add the effective date of the entry into the database (PRIVATE).

        record - a SeqRecord object with an annotated date
        bioentry_id - corresponding database identifier
        """
    date = record.annotations.get('date', strftime('%d-%b-%Y', gmtime()).upper())
    if isinstance(date, list):
        date = date[0]
    annotation_tags_id = self._get_ontology_id('Annotation Tags')
    date_id = self._get_term_id('date_changed', annotation_tags_id)
    sql = 'INSERT INTO bioentry_qualifier_value (bioentry_id, term_id, value, "rank") VALUES (%s, %s, %s, 1)'
    self.adaptor.execute(sql, (bioentry_id, date_id, date))