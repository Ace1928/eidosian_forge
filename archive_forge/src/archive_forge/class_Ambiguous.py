import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
class Ambiguous(AbstractCut):
    """Implement methods for enzymes that produce variable overhangs.

    Typical example : BstXI -> CCAN_NNNN^NTGG
                      The overhang can be any sequence of 4 bases.

    Notes:
        Blunt enzymes are always defined. Even if their site is GGATCCNNN^_N
        Their overhang is always the same : blunt!

    Internal use only. Not meant to be instantiated.

    """

    @classmethod
    def _drop(cls):
        """Remove cuts that are outsite of the sequence (PRIVATE).

        For internal use only.

        Drop the site that are situated outside the sequence in linear
        sequence. Modify the index for site in circular sequences.
        """
        length = len(cls.dna)
        drop = itertools.dropwhile
        take = itertools.takewhile
        if cls.dna.is_linear():
            cls.results = list(drop(lambda x: x <= 1, cls.results))
            cls.results = list(take(lambda x: x <= length, cls.results))
        else:
            for index, location in enumerate(cls.results):
                if location < 1:
                    cls.results[index] += length
                else:
                    break
            for index, location in enumerate(cls.results[::-1]):
                if location > length:
                    cls.results[-(index + 1)] -= length
                else:
                    break

    @classmethod
    def is_defined(cls):
        """Return if recognition sequence and cut are defined.

        True if the sequence recognised and cut is constant,
        i.e. the recognition site is not degenerated AND the enzyme cut inside
        the site.

        Related methods:

        - RE.is_ambiguous()
        - RE.is_unknown()

        """
        return False

    @classmethod
    def is_ambiguous(cls):
        """Return if recognition sequence and cut may be ambiguous.

        True if the sequence recognised and cut is ambiguous,
        i.e. the recognition site is degenerated AND/OR the enzyme cut outside
        the site.

        Related methods:

        - RE.is_defined()
        - RE.is_unknown()

        """
        return True

    @classmethod
    def is_unknown(cls):
        """Return if recognition sequence is unknown.

        True if the sequence is unknown,
        i.e. the recognition site has not been characterised yet.

        Related methods:

        - RE.is_defined()
        - RE.is_ambiguous()

        """
        return False

    @classmethod
    def _mod2(cls, other):
        """Test if other enzyme produces compatible ends for enzyme (PRIVATE).

        For internal use only.

        Test for the compatibility of restriction ending of RE and other.
        """
        if len(cls.ovhgseq) != len(other.ovhgseq):
            return False
        else:
            se = cls.ovhgseq
            for base in se:
                if base in 'ATCG':
                    pass
                if base in 'N':
                    se = '.'.join(se.split('N'))
                if base in 'RYWMSKHDBV':
                    expand = '[' + matching[base] + ']'
                    se = expand.join(se.split(base))
            if re.match(se, other.ovhgseq):
                return True
            else:
                return False

    @classmethod
    def elucidate(cls):
        """Return a string representing the recognition site and cuttings.

        Return a representation of the site with the cut on the (+) strand
        represented as '^' and the cut on the (-) strand as '_'.
        ie:

        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI
        >>> EcoRI.elucidate()   # 5' overhang
        'G^AATT_C'
        >>> KpnI.elucidate()    # 3' overhang
        'G_GTAC^C'
        >>> EcoRV.elucidate()   # blunt
        'GAT^_ATC'
        >>> SnaI.elucidate()     # NotDefined, cut profile unknown.
        '? GTATAC ?'
        >>>

        """
        f5 = cls.fst5
        f3 = cls.fst3
        length = len(cls)
        site = cls.site
        if cls.cut_twice():
            re = 'cut twice, not yet implemented sorry.'
        elif cls.is_5overhang():
            if f3 == f5 == 0:
                re = 'N^' + site + '_N'
            elif 0 <= f5 <= length and 0 <= f3 + length <= length:
                re = site[:f5] + '^' + site[f5:f3] + '_' + site[f3:]
            elif 0 <= f5 <= length:
                re = site[:f5] + '^' + site[f5:] + f3 * 'N' + '_N'
            elif 0 <= f3 + length <= length:
                re = 'N^' + abs(f5) * 'N' + site[:f3] + '_' + site[f3:]
            elif f3 + length < 0:
                re = 'N^' + abs(f5) * 'N' + '_' + abs(length + f3) * 'N' + site
            elif f5 > length:
                re = site + (f5 - length) * 'N' + '^' + (length + f3 - f5) * 'N' + '_N'
            else:
                re = 'N^' + abs(f5) * 'N' + site + f3 * 'N' + '_N'
        elif cls.is_blunt():
            if f5 < 0:
                re = 'N^_' + abs(f5) * 'N' + site
            elif f5 > length:
                re = site + (f5 - length) * 'N' + '^_N'
            else:
                raise ValueError('%s.easyrepr() : error f5=%i' % (cls.name, f5))
        elif f3 == 0:
            if f5 == 0:
                re = 'N_' + site + '^N'
            else:
                re = site + '_' + (f5 - length) * 'N' + '^N'
        elif 0 < f3 + length <= length and 0 <= f5 <= length:
            re = site[:f3] + '_' + site[f3:f5] + '^' + site[f5:]
        elif 0 < f3 + length <= length:
            re = site[:f3] + '_' + site[f3:] + (f5 - length) * 'N' + '^N'
        elif 0 <= f5 <= length:
            re = 'N_' + 'N' * (f3 + length) + site[:f5] + '^' + site[f5:]
        elif f3 > 0:
            re = site + f3 * 'N' + '_' + (f5 - f3 - length) * 'N' + '^N'
        elif f5 < 0:
            re = 'N_' + abs(f3 - f5 + length) * 'N' + '^' + abs(f5) * 'N' + site
        else:
            re = 'N_' + abs(f3 + length) * 'N' + site + (f5 - length) * 'N' + '^N'
        return re