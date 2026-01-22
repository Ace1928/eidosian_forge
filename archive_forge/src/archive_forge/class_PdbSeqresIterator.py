import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
class PdbSeqresIterator(SequenceIterator):
    """Parser for PDB files."""

    def __init__(self, source):
        """Return SeqRecord objects for each chain in a PDB file.

        Arguments:
         - source - input stream opened in text mode, or a path to a file

        The sequences are derived from the SEQRES lines in the
        PDB file header, not the atoms of the 3D structure.

        Specifically, these PDB records are handled: DBREF, DBREF1, DBREF2, SEQADV, SEQRES, MODRES

        See: http://www.wwpdb.org/documentation/format23/sect3.html

        This gets called internally via Bio.SeqIO for the SEQRES based interpretation
        of the PDB file format:

        >>> from Bio import SeqIO
        >>> for record in SeqIO.parse("PDB/1A8O.pdb", "pdb-seqres"):
        ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
        ...     print(record.dbxrefs)
        ...
        Record id 1A8O:A, chain A
        ['UNP:P12497', 'UNP:POL_HV1N5']

        Equivalently,

        >>> with open("PDB/1A8O.pdb") as handle:
        ...     for record in PdbSeqresIterator(handle):
        ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
        ...         print(record.dbxrefs)
        ...
        Record id 1A8O:A, chain A
        ['UNP:P12497', 'UNP:POL_HV1N5']

        Note the chain is recorded in the annotations dictionary, and any PDB DBREF
        lines are recorded in the database cross-references list.
        """
        super().__init__(source, mode='t', fmt='PDB')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Iterate over the records in the PDB file."""
        chains = collections.defaultdict(list)
        metadata = collections.defaultdict(list)
        rec_name = None
        for line in handle:
            rec_name = line[0:6].strip()
            if rec_name == 'SEQRES':
                chn_id = line[11]
                residues = [_res2aacode(res) for res in line[19:].split()]
                chains[chn_id].extend(residues)
            elif rec_name == 'DBREF':
                pdb_id = line[7:11]
                chn_id = line[12]
                database = line[26:32].strip()
                db_acc = line[33:41].strip()
                db_id_code = line[42:54].strip()
                metadata[chn_id].append({'pdb_id': pdb_id, 'database': database, 'db_acc': db_acc, 'db_id_code': db_id_code})
            elif rec_name == 'DBREF1':
                pdb_id = line[7:11]
                chn_id = line[12]
                database = line[26:32].strip()
                db_id_code = line[47:67].strip()
            elif rec_name == 'DBREF2':
                if pdb_id != line[7:11] or chn_id != line[12]:
                    raise ValueError('DBREF2 identifiers do not match')
                db_acc = line[18:40].strip()
                metadata[chn_id].append({'pdb_id': pdb_id, 'database': database, 'db_acc': db_acc, 'db_id_code': db_id_code})
        if rec_name is None:
            raise ValueError('Empty file.')
        for chn_id, residues in sorted(chains.items()):
            record = SeqRecord(Seq(''.join(residues)))
            record.annotations = {'chain': chn_id}
            record.annotations['molecule_type'] = 'protein'
            if chn_id in metadata:
                m = metadata[chn_id][0]
                record.id = record.name = f'{m['pdb_id']}:{chn_id}'
                record.description = f'{m['database']}:{m['db_acc']} {m['db_id_code']}'
                for melem in metadata[chn_id]:
                    record.dbxrefs.extend([f'{melem['database']}:{melem['db_acc']}', f'{melem['database']}:{melem['db_id_code']}'])
            else:
                record.id = chn_id
            yield record