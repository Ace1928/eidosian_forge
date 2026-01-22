import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def CifSeqresIterator(source):
    """Return SeqRecord objects for each chain in an mmCIF file.

    Argument source is a file-like object or a path to a file.

    The sequences are derived from the _entity_poly_seq entries in the mmCIF
    file, not the atoms of the 3D structure.

    Specifically, these mmCIF records are handled: _pdbx_poly_seq_scheme and
    _struct_ref_seq. The _pdbx_poly_seq records contain sequence information,
    and the _struct_ref_seq records contain database cross-references.

    See:
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/pdbx_poly_seq_scheme.html
    and
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/struct_ref_seq.html

    This gets called internally via Bio.SeqIO for the sequence-based
    interpretation of the mmCIF file format:

    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("PDB/1A8O.cif", "cif-seqres"):
    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...     print(record.dbxrefs)
    ...
    Record id 1A8O:A, chain A
    ['UNP:P12497', 'UNP:POL_HV1N5']

    Equivalently,

    >>> with open("PDB/1A8O.cif") as handle:
    ...     for record in CifSeqresIterator(handle):
    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    ...         print(record.dbxrefs)
    ...
    Record id 1A8O:A, chain A
    ['UNP:P12497', 'UNP:POL_HV1N5']

    Note the chain is recorded in the annotations dictionary, and any mmCIF
    _struct_ref_seq entries are recorded in the database cross-references list.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    chains = collections.defaultdict(list)
    metadata = collections.defaultdict(list)
    records = MMCIF2Dict(source)
    for field in PDBX_POLY_SEQ_SCHEME_FIELDS + STRUCT_REF_SEQ_FIELDS + STRUCT_REF_FIELDS:
        if field not in records:
            records[field] = []
        elif not isinstance(records[field], list):
            records[field] = [records[field]]
    for asym_id, mon_id in zip(records['_pdbx_poly_seq_scheme.asym_id'], records['_pdbx_poly_seq_scheme.mon_id']):
        mon_id_1l = _res2aacode(mon_id)
        chains[asym_id].append(mon_id_1l)
    struct_refs = {}
    for ref_id, db_name, db_code, db_acc in zip(records['_struct_ref.id'], records['_struct_ref.db_name'], records['_struct_ref.db_code'], records['_struct_ref.pdbx_db_accession']):
        struct_refs[ref_id] = {'database': db_name, 'db_id_code': db_code, 'db_acc': db_acc}
    for ref_id, pdb_id, chain_id in zip(records['_struct_ref_seq.ref_id'], records['_struct_ref_seq.pdbx_PDB_id_code'], records['_struct_ref_seq.pdbx_strand_id']):
        struct_ref = struct_refs[ref_id]
        metadata[chain_id].append({'pdb_id': pdb_id})
        metadata[chain_id][-1].update(struct_ref)
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