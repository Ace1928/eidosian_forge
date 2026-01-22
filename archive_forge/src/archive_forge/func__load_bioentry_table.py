from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_bioentry_table(self, record):
    """Fill the bioentry table with sequence information (PRIVATE).

        Arguments:
         - record - SeqRecord object to add to the database.

        """
    if record.id.count('.') == 1:
        accession, version = record.id.split('.')
        try:
            version = int(version)
        except ValueError:
            accession = record.id
            version = 0
    else:
        accession = record.id
        version = 0
    if 'accessions' in record.annotations and isinstance(record.annotations['accessions'], list) and record.annotations['accessions']:
        accession = record.annotations['accessions'][0]
    taxon_id = self._get_taxon_id(record)
    if 'gi' in record.annotations:
        identifier = record.annotations['gi']
    else:
        identifier = record.id
    description = getattr(record, 'description', None)
    division = record.annotations.get('data_file_division')
    sql = '\n        INSERT INTO bioentry (\n         biodatabase_id,\n         taxon_id,\n         name,\n         accession,\n         identifier,\n         division,\n         description,\n         version)\n        VALUES (\n         %s,\n         %s,\n         %s,\n         %s,\n         %s,\n         %s,\n         %s,\n         %s)'
    self.adaptor.execute(sql, (self.dbid, taxon_id, record.name, accession, identifier, division, description, version))
    return self.adaptor.last_id('bioentry')