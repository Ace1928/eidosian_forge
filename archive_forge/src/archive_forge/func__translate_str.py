import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
def _translate_str(sequence, table, stop_symbol='*', to_stop=False, cds=False, pos_stop='X', gap=None):
    """Translate nucleotide string into a protein string (PRIVATE).

    Arguments:
     - sequence - a string
     - table - Which codon table to use?  This can be either a name (string),
       an NCBI identifier (integer), or a CodonTable object (useful for
       non-standard genetic codes).  This defaults to the "Standard" table.
     - stop_symbol - a single character string, what to use for terminators.
     - to_stop - boolean, should translation terminate at the first
       in frame stop codon?  If there is no in-frame stop codon
       then translation continues to the end.
     - pos_stop - a single character string for a possible stop codon
       (e.g. TAN or NNN)
     - cds - Boolean, indicates this is a complete CDS.  If True, this
       checks the sequence starts with a valid alternative start
       codon (which will be translated as methionine, M), that the
       sequence length is a multiple of three, and that there is a
       single in frame stop codon at the end (this will be excluded
       from the protein sequence, regardless of the to_stop option).
       If these tests fail, an exception is raised.
     - gap - Single character string to denote symbol used for gaps.
       Defaults to None.

    Returns a string.

    e.g.

    >>> from Bio.Data import CodonTable
    >>> table = CodonTable.ambiguous_dna_by_id[1]
    >>> _translate_str("AAA", table)
    'K'
    >>> _translate_str("TAR", table)
    '*'
    >>> _translate_str("TAN", table)
    'X'
    >>> _translate_str("TAN", table, pos_stop="@")
    '@'
    >>> _translate_str("TA?", table)
    Traceback (most recent call last):
       ...
    Bio.Data.CodonTable.TranslationError: Codon 'TA?' is invalid

    In a change to older versions of Biopython, partial codons are now
    always regarded as an error (previously only checked if cds=True)
    and will trigger a warning (likely to become an exception in a
    future release).

    If **cds=True**, the start and stop codons are checked, and the start
    codon will be translated at methionine. The sequence must be an
    while number of codons.

    >>> _translate_str("ATGCCCTAG", table, cds=True)
    'MP'
    >>> _translate_str("AAACCCTAG", table, cds=True)
    Traceback (most recent call last):
       ...
    Bio.Data.CodonTable.TranslationError: First codon 'AAA' is not a start codon
    >>> _translate_str("ATGCCCTAGCCCTAG", table, cds=True)
    Traceback (most recent call last):
       ...
    Bio.Data.CodonTable.TranslationError: Extra in frame stop codon 'TAG' found.
    """
    try:
        table_id = int(table)
    except ValueError:
        try:
            codon_table = CodonTable.ambiguous_generic_by_name[table]
        except KeyError:
            if isinstance(table, str):
                raise ValueError("The Bio.Seq translate methods and function DO NOT take a character string mapping table like the python string object's translate method. Use str(my_seq).translate(...) instead.") from None
            else:
                raise TypeError('table argument must be integer or string') from None
    except (AttributeError, TypeError):
        if isinstance(table, CodonTable.CodonTable):
            codon_table = table
        else:
            raise ValueError('Bad table argument') from None
    else:
        codon_table = CodonTable.ambiguous_generic_by_id[table_id]
    sequence = sequence.upper()
    amino_acids = []
    forward_table = codon_table.forward_table
    stop_codons = codon_table.stop_codons
    if codon_table.nucleotide_alphabet is not None:
        valid_letters = set(codon_table.nucleotide_alphabet.upper())
    else:
        valid_letters = set(IUPACData.ambiguous_dna_letters.upper() + IUPACData.ambiguous_rna_letters.upper())
    n = len(sequence)
    dual_coding = [c for c in stop_codons if c in forward_table]
    if dual_coding:
        c = dual_coding[0]
        if to_stop:
            raise ValueError(f"You cannot use 'to_stop=True' with this table as it contains {len(dual_coding)} codon(s) which can be both STOP and an amino acid (e.g. '{c}' -> '{forward_table[c]}' or STOP).")
        warnings.warn(f"This table contains {len(dual_coding)} codon(s) which code(s) for both STOP and an amino acid (e.g. '{c}' -> '{forward_table[c]}' or STOP). Such codons will be translated as amino acid.", BiopythonWarning)
    if cds:
        if str(sequence[:3]).upper() not in codon_table.start_codons:
            raise CodonTable.TranslationError(f"First codon '{sequence[:3]}' is not a start codon")
        if n % 3 != 0:
            raise CodonTable.TranslationError(f'Sequence length {n} is not a multiple of three')
        if str(sequence[-3:]).upper() not in stop_codons:
            raise CodonTable.TranslationError(f"Final codon '{sequence[-3:]}' is not a stop codon")
        sequence = sequence[3:-3]
        n -= 6
        amino_acids = ['M']
    elif n % 3 != 0:
        warnings.warn('Partial codon, len(sequence) not a multiple of three. Explicitly trim the sequence or add trailing N before translation. This may become an error in future.', BiopythonWarning)
    if gap is not None:
        if not isinstance(gap, str):
            raise TypeError('Gap character should be a single character string.')
        elif len(gap) > 1:
            raise ValueError('Gap character should be a single character string.')
    for i in range(0, n - n % 3, 3):
        codon = sequence[i:i + 3]
        try:
            amino_acids.append(forward_table[codon])
        except (KeyError, CodonTable.TranslationError):
            if codon in codon_table.stop_codons:
                if cds:
                    raise CodonTable.TranslationError(f"Extra in frame stop codon '{codon}' found.") from None
                if to_stop:
                    break
                amino_acids.append(stop_symbol)
            elif valid_letters.issuperset(set(codon)):
                amino_acids.append(pos_stop)
            elif gap is not None and codon == gap * 3:
                amino_acids.append(gap)
            else:
                raise CodonTable.TranslationError(f"Codon '{codon}' is invalid") from None
    return ''.join(amino_acids)