import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def Tm_GC(seq, check=True, strict=True, valueset=7, userset=None, Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=0, mismatch=True):
    """Return the Tm using empirical formulas based on GC content.

    General format: Tm = A + B(%GC) - C/N + salt correction - D(%mismatch)

    A, B, C, D: empirical constants, N: primer length
    D (amount of decrease in Tm per % mismatch) is often 1, but sometimes other
    values have been used (0.6-1.5). Use 'X' to indicate the mismatch position
    in the sequence. Note that this mismatch correction is a rough estimate.

    >>> from Bio.SeqUtils import MeltingTemp as mt
    >>> print("%0.2f" % mt.Tm_GC('CTGCTGATXGCACGAGGTTATGG', valueset=2))
    69.20

    Arguments:
     - valueset: A few often cited variants are included:

        1. Tm = 69.3 + 0.41(%GC) - 650/N
           (Marmur & Doty 1962, J Mol Biol 5: 109-118; Chester & Marshak 1993),
           Anal Biochem 209: 284-290)
        2. Tm = 81.5 + 0.41(%GC) - 675/N - %mismatch
           'QuikChange' formula. Recommended (by the manufacturer) for the
           design of primers for QuikChange mutagenesis.
        3. Tm = 81.5 + 0.41(%GC) - 675/N + 16.6 x log[Na+]
           (Marmur & Doty 1962, J Mol Biol 5: 109-118; Schildkraut & Lifson
           1965, Biopolymers 3: 195-208)
        4. Tm = 81.5 + 0.41(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x
           [Na+])) - %mismatch
           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). This is the
           standard formula in approximative mode of MELTING 4.3.
        5. Tm = 78 + 0.7(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x [Na+]))
           - %mismatch
           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). For RNA.
        6. Tm = 67 + 0.8(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x [Na+]))
           - %mismatch
           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). For RNA/DNA
           hybrids.
        7. Tm = 81.5 + 0.41(%GC) - 600/N + 16.6 x log[Na+]
           Used by Primer3Plus to calculate the product Tm. Default set.
        8. Tm = 77.1 + 0.41(%GC) - 528/N + 11.7 x log[Na+]
           (von Ahsen et al. 2001, Clin Chem 47: 1956-1961). Recommended 'as a
           tradeoff between accuracy and ease of use'.

     - userset: Tuple of four values for A, B, C, and D. Usersets override
       valuesets.
     - Na, K, Tris, Mg, dNTPs: Concentration of the respective ions [mM]. If
       any of K, Tris, Mg and dNTPS is non-zero, a 'sodium-equivalent'
       concentration is calculated and used for salt correction (von Ahsen et
       al., 2001).
     - saltcorr: Type of salt correction (see method salt_correction).
       Default=0. 0 or None means no salt correction.
     - mismatch: If 'True' (default) every 'X' in the sequence is counted as
       mismatch.

    """
    if saltcorr == 5:
        raise ValueError('salt-correction method 5 not applicable to Tm_GC')
    seq = str(seq)
    if check:
        seq = _check(seq, 'Tm_GC')
    if strict and any((x in seq for x in 'KMNRYBVDH')):
        raise ValueError("ambiguous bases B, D, H, K, M, N, R, V, Y not allowed when 'strict=True'")
    percent_gc = SeqUtils.gc_fraction(seq, 'weighted') * 100
    if mismatch:
        percent_gc -= seq.count('X') * 50.0 / len(seq)
    if userset:
        A, B, C, D = userset
    else:
        if valueset == 1:
            A, B, C, D = (69.3, 0.41, 650, 1)
            saltcorr = 0
        if valueset == 2:
            A, B, C, D = (81.5, 0.41, 675, 1)
            saltcorr = 0
        if valueset == 3:
            A, B, C, D = (81.5, 0.41, 675, 1)
            saltcorr = 1
        if valueset == 4:
            A, B, C, D = (81.5, 0.41, 500, 1)
            saltcorr = 2
        if valueset == 5:
            A, B, C, D = (78.0, 0.7, 500, 1)
            saltcorr = 2
        if valueset == 6:
            A, B, C, D = (67.0, 0.8, 500, 1)
            saltcorr = 2
        if valueset == 7:
            A, B, C, D = (81.5, 0.41, 600, 1)
            saltcorr = 1
        if valueset == 8:
            A, B, C, D = (77.1, 0.41, 528, 1)
            saltcorr = 4
    if valueset > 8:
        raise ValueError("allowed values for parameter 'valueset' are 0-8.")
    melting_temp = A + B * percent_gc - C / len(seq)
    if saltcorr:
        melting_temp += salt_correction(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, seq=seq, method=saltcorr)
    if mismatch:
        melting_temp -= D * (seq.count('X') * 100.0 / len(seq))
    return melting_temp