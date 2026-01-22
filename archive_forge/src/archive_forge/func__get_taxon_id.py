from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_taxon_id(self, record):
    """Get the taxon id for this record (PRIVATE).

        Arguments:
         - record - a SeqRecord object

        This searches the taxon/taxon_name tables using the
        NCBI taxon ID, scientific name and common name to find
        the matching taxon table entry's id.

        If the species isn't in the taxon table, and we have at
        least the NCBI taxon ID, scientific name or common name,
        at least a minimal stub entry is created in the table.

        Returns the taxon id (database key for the taxon table,
        not an NCBI taxon ID), or None if the taxonomy information
        is missing.

        See also the BioSQL script load_ncbi_taxonomy.pl which
        will populate and update the taxon/taxon_name tables
        with the latest information from the NCBI.
        """
    ncbi_taxon_id = None
    if 'ncbi_taxid' in record.annotations:
        if isinstance(record.annotations['ncbi_taxid'], list):
            if len(record.annotations['ncbi_taxid']) == 1:
                ncbi_taxon_id = record.annotations['ncbi_taxid'][0]
        else:
            ncbi_taxon_id = record.annotations['ncbi_taxid']
    if not ncbi_taxon_id:
        for f in record.features:
            if f.type == 'source':
                quals = getattr(f, 'qualifiers', {})
                if 'db_xref' in quals:
                    for db_xref in f.qualifiers['db_xref']:
                        if db_xref.startswith('taxon:'):
                            ncbi_taxon_id = int(db_xref[6:])
                            break
            if ncbi_taxon_id:
                break
    try:
        scientific_name = record.annotations['organism'][:255]
    except KeyError:
        scientific_name = None
    try:
        common_name = record.annotations['source'][:255]
    except KeyError:
        common_name = None
    if ncbi_taxon_id:
        return self._get_taxon_id_from_ncbi_taxon_id(ncbi_taxon_id, scientific_name, common_name)
    if not common_name and (not scientific_name):
        return None
    if scientific_name:
        taxa = self.adaptor.execute_and_fetch_col0("SELECT taxon_id FROM taxon_name WHERE name_class = 'scientific name' AND name = %s", (scientific_name,))
        if taxa:
            return taxa[0]
    if common_name:
        taxa = self.adaptor.execute_and_fetch_col0('SELECT DISTINCT taxon_id FROM taxon_name WHERE name = %s', (common_name,))
        if len(taxa) > 1:
            raise ValueError('Taxa: %d species have name %r' % (len(taxa), common_name))
        if taxa:
            return taxa[0]
    lineage = []
    for c in record.annotations.get('taxonomy', []):
        lineage.append([None, None, c])
    if lineage:
        lineage[-1][1] = 'genus'
    lineage.append([None, 'species', record.annotations['organism']])
    if 'subspecies' in record.annotations:
        lineage.append([None, 'subspecies', record.annotations['subspecies']])
    if 'variant' in record.annotations:
        lineage.append([None, 'varietas', record.annotations['variant']])
    lineage[-1][0] = ncbi_taxon_id
    left_value = self.adaptor.execute_one('SELECT MAX(left_value) FROM taxon')[0]
    if not left_value:
        left_value = 0
    left_value += 1
    right_start_value = self.adaptor.execute_one('SELECT MAX(right_value) FROM taxon')[0]
    if not right_start_value:
        right_start_value = 0
    right_value = right_start_value + 2 * len(lineage) - 1
    parent_taxon_id = None
    for taxon in lineage:
        self.adaptor.execute('INSERT INTO taxon(parent_taxon_id, ncbi_taxon_id, node_rank, left_value, right_value) VALUES (%s, %s, %s, %s, %s)', (parent_taxon_id, taxon[0], taxon[1], left_value, right_value))
        taxon_id = self.adaptor.last_id('taxon')
        self.adaptor.execute("INSERT INTO taxon_name(taxon_id, name, name_class)VALUES (%s, %s, 'scientific name')", (taxon_id, taxon[2][:255]))
        left_value += 1
        right_value -= 1
        parent_taxon_id = taxon_id
    if common_name:
        self.adaptor.execute("INSERT INTO taxon_name(taxon_id, name, name_class)VALUES (%s, %s, 'common name')", (taxon_id, common_name))
    return taxon_id