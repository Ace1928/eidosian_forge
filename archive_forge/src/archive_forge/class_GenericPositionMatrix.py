import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
class GenericPositionMatrix(dict):
    """Base class for the support of position matrix operations."""

    def __init__(self, alphabet, values):
        """Initialize the class."""
        self.length = None
        for letter in alphabet:
            if self.length is None:
                self.length = len(values[letter])
            elif self.length != len(values[letter]):
                raise Exception('data has inconsistent lengths')
            self[letter] = list(values[letter])
        self.alphabet = alphabet

    def __str__(self):
        """Return a string containing nucleotides and counts of the alphabet in the Matrix."""
        words = ['%6d' % i for i in range(self.length)]
        line = '   ' + ' '.join(words)
        lines = [line]
        for letter in self.alphabet:
            words = ['%6.2f' % value for value in self[letter]]
            line = '%c: ' % letter + ' '.join(words)
            lines.append(line)
        text = '\n'.join(lines) + '\n'
        return text

    def __getitem__(self, key):
        """Return the position matrix of index key."""
        if isinstance(key, tuple):
            if len(key) == 2:
                key1, key2 = key
                if isinstance(key1, slice):
                    start1, stop1, stride1 = key1.indices(len(self.alphabet))
                    indices1 = range(start1, stop1, stride1)
                    letters1 = [self.alphabet[i] for i in indices1]
                    dim1 = 2
                elif isinstance(key1, numbers.Integral):
                    letter1 = self.alphabet[key1]
                    dim1 = 1
                elif isinstance(key1, tuple):
                    letters1 = [self.alphabet[i] for i in key1]
                    dim1 = 2
                elif isinstance(key1, str):
                    if len(key1) == 1:
                        letter1 = key1
                        dim1 = 1
                    else:
                        raise KeyError(key1)
                else:
                    raise KeyError('Cannot understand key %s' % key1)
                if isinstance(key2, slice):
                    start2, stop2, stride2 = key2.indices(self.length)
                    indices2 = range(start2, stop2, stride2)
                    dim2 = 2
                elif isinstance(key2, numbers.Integral):
                    index2 = key2
                    dim2 = 1
                else:
                    raise KeyError('Cannot understand key %s' % key2)
                if dim1 == 1 and dim2 == 1:
                    return dict.__getitem__(self, letter1)[index2]
                elif dim1 == 1 and dim2 == 2:
                    values = dict.__getitem__(self, letter1)
                    return tuple((values[index2] for index2 in indices2))
                elif dim1 == 2 and dim2 == 1:
                    d = {}
                    for letter1 in letters1:
                        d[letter1] = dict.__getitem__(self, letter1)[index2]
                    return d
                else:
                    d = {}
                    for letter1 in letters1:
                        values = dict.__getitem__(self, letter1)
                        d[letter1] = [values[_] for _ in indices2]
                    if sorted(letters1) == self.alphabet:
                        return self.__class__(self.alphabet, d)
                    else:
                        return d
            elif len(key) == 1:
                key = key[0]
            else:
                raise KeyError('keys should be 1- or 2-dimensional')
        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self.alphabet))
            indices = range(start, stop, stride)
            letters = [self.alphabet[i] for i in indices]
            dim = 2
        elif isinstance(key, numbers.Integral):
            letter = self.alphabet[key]
            dim = 1
        elif isinstance(key, tuple):
            letters = [self.alphabet[i] for i in key]
            dim = 2
        elif isinstance(key, str):
            if len(key) == 1:
                letter = key
                dim = 1
            else:
                raise KeyError(key)
        else:
            raise KeyError('Cannot understand key %s' % key)
        if dim == 1:
            return dict.__getitem__(self, letter)
        elif dim == 2:
            d = {}
            for letter in letters:
                d[letter] = dict.__getitem__(self, letter)
            return d
        else:
            raise RuntimeError('Should not get here')

    @property
    def consensus(self):
        """Return the consensus sequence."""
        sequence = ''
        for i in range(self.length):
            maximum = -math.inf
            for letter in self.alphabet:
                count = self[letter][i]
                if count > maximum:
                    maximum = count
                    sequence_letter = letter
            sequence += sequence_letter
        return Seq(sequence)

    @property
    def anticonsensus(self):
        """Return the anticonsensus sequence."""
        sequence = ''
        for i in range(self.length):
            minimum = math.inf
            for letter in self.alphabet:
                count = self[letter][i]
                if count < minimum:
                    minimum = count
                    sequence_letter = letter
            sequence += sequence_letter
        return Seq(sequence)

    @property
    def degenerate_consensus(self):
        """Return the degenerate consensus sequence."""
        degenerate_nucleotide = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'AC': 'M', 'AG': 'R', 'AT': 'W', 'CG': 'S', 'CT': 'Y', 'GT': 'K', 'ACG': 'V', 'ACT': 'H', 'AGT': 'D', 'CGT': 'B', 'ACGT': 'N'}
        sequence = ''
        for i in range(self.length):

            def get(nucleotide):
                return self[nucleotide][i]
            nucleotides = sorted(self, key=get, reverse=True)
            counts = [self[c][i] for c in nucleotides]
            if counts[0] > sum(counts[1:]) and counts[0] > 2 * counts[1]:
                key = nucleotides[0]
            elif 4 * sum(counts[:2]) > 3 * sum(counts):
                key = ''.join(sorted(nucleotides[:2]))
            elif counts[3] == 0:
                key = ''.join(sorted(nucleotides[:3]))
            else:
                key = 'ACGT'
            nucleotide = degenerate_nucleotide.get(key, key)
            sequence += nucleotide
        return Seq(sequence)

    def calculate_consensus(self, substitution_matrix=None, plurality=None, identity=0, setcase=None):
        """Return the consensus sequence (as a string) for the given parameters.

        This function largely follows the conventions of the EMBOSS `cons` tool.

        Arguments:
         - substitution_matrix - the scoring matrix used when comparing
           sequences. By default, it is None, in which case we simply count the
           frequency of each letter.
           Instead of the default value, you can use the substitution matrices
           available in Bio.Align.substitution_matrices. Common choices are
           BLOSUM62 (also known as EBLOSUM62) for protein, and NUC.4.4 (also
           known as EDNAFULL) for nucleotides. NOTE: This has not yet been
           implemented.
         - plurality           - threshold value for the number of positive
           matches, divided by the total count in a column, required to reach
           consensus. If substitution_matrix is None, then this argument must
           be None, and is ignored; a ValueError is raised otherwise. If
           substitution_matrix is not None, then the default value of the
           plurality is 0.5.
         - identity            - number of identities, divided by the total
           count in a column, required to define a consensus value. If the
           number of identities is less than identity * total count in a column,
           then the undefined character ('N' for nucleotides and 'X' for amino
           acid sequences) is used in the consensus sequence. If identity is
           1.0, then only columns of identical letters contribute to the
           consensus. Default value is zero.
         - setcase             - threshold for the positive matches, divided by
           the total count in a column, above which the consensus is is
           upper-case and below which the consensus is in lower-case. By
           default, this is equal to 0.5.
        """
        alphabet = self.alphabet
        if set(alphabet).union('ACGTUN-') == set('ACGTUN-'):
            undefined = 'N'
        else:
            undefined = 'X'
        if substitution_matrix is None:
            if plurality is not None:
                raise ValueError('plurality must be None if substitution_matrix is None')
            sequence = ''
            for i in range(self.length):
                maximum = 0
                total = 0
                for letter in alphabet:
                    count = self[letter][i]
                    total += count
                    if count > maximum:
                        maximum = count
                        consensus_letter = letter
                if maximum < identity * total:
                    consensus_letter = undefined
                else:
                    if setcase is None:
                        setcase_threshold = total / 2
                    else:
                        setcase_threshold = setcase * total
                    if maximum <= setcase_threshold:
                        consensus_letter = consensus_letter.lower()
                sequence += consensus_letter
        else:
            raise NotImplementedError('calculate_consensus currently only supports substitution_matrix=None')
        return sequence

    @property
    def gc_content(self):
        """Compute the fraction GC content."""
        alphabet = self.alphabet
        gc_total = 0.0
        total = 0.0
        for i in range(self.length):
            for letter in alphabet:
                if letter in 'CG':
                    gc_total += self[letter][i]
                total += self[letter][i]
        return gc_total / total

    def reverse_complement(self):
        """Compute reverse complement."""
        values = {}
        if self.alphabet == 'ACGU':
            values['A'] = self['U'][::-1]
            values['U'] = self['A'][::-1]
        else:
            values['A'] = self['T'][::-1]
            values['T'] = self['A'][::-1]
        values['G'] = self['C'][::-1]
        values['C'] = self['G'][::-1]
        alphabet = self.alphabet
        return self.__class__(alphabet, values)