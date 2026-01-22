import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def AtomIterator(pdb_id, structure):
    """Return SeqRecords from Structure objects.

    Base function for sequence parsers that read structures Bio.PDB parsers.

    Once a parser from Bio.PDB has been used to load a structure into a
    Bio.PDB.Structure.Structure object, there is no difference in how the
    sequence parser interprets the residue sequence. The functions in this
    module may be used by SeqIO modules wishing to parse sequences from lists
    of residues.

    Calling functions must pass a Bio.PDB.Structure.Structure object.


    See Bio.SeqIO.PdbIO.PdbAtomIterator and Bio.SeqIO.PdbIO.CifAtomIterator for
    details.
    """
    model = structure[0]
    for chn_id, chain in sorted(model.child_dict.items()):
        residues = [res for res in chain.get_unpacked_list() if _res2aacode(res.get_resname().upper()) != 'X']
        if not residues:
            continue
        gaps = []
        rnumbers = [r.id[1] for r in residues]
        for i, rnum in enumerate(rnumbers[:-1]):
            if rnumbers[i + 1] != rnum + 1 and rnumbers[i + 1] != rnum:
                gaps.append((i + 1, rnum, rnumbers[i + 1]))
        if gaps:
            res_out = []
            prev_idx = 0
            for i, pregap, postgap in gaps:
                if postgap > pregap:
                    gapsize = postgap - pregap - 1
                    res_out.extend((_res2aacode(x) for x in residues[prev_idx:i]))
                    prev_idx = i
                    res_out.append('X' * gapsize)
                else:
                    warnings.warn('Ignoring out-of-order residues after a gap', BiopythonParserWarning)
                    res_out.extend((_res2aacode(x) for x in residues[prev_idx:i]))
                    break
            else:
                res_out.extend((_res2aacode(x) for x in residues[prev_idx:]))
        else:
            res_out = [_res2aacode(x) for x in residues]
        record_id = f'{pdb_id}:{chn_id}'
        record = SeqRecord(Seq(''.join(res_out)), id=record_id, description=record_id)
        record.annotations['molecule_type'] = 'protein'
        record.annotations['model'] = model.id
        record.annotations['chain'] = chain.id
        record.annotations['start'] = int(rnumbers[0])
        record.annotations['end'] = int(rnumbers[-1])
        yield record