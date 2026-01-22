import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
class PositionSpecificScoringMatrix(GenericPositionMatrix):
    """Class for the support of Position Specific Scoring Matrix calculations."""

    def calculate(self, sequence):
        """Return the PWM score for a given sequence for all positions.

        Notes:
         - the sequence can only be a DNA sequence
         - the search is performed only on one strand
         - if the sequence and the motif have the same length, a single
           number is returned
         - otherwise, the result is a one-dimensional numpy array

        """
        if sorted(self.alphabet) != ['A', 'C', 'G', 'T']:
            raise ValueError('PSSM has wrong alphabet: %s - Use only with DNA motifs' % self.alphabet)
        try:
            sequence = bytes(sequence)
        except TypeError:
            try:
                sequence = bytes(sequence, 'ASCII')
            except TypeError:
                raise ValueError('sequence should be a Seq, MutableSeq, string, or bytes-like object') from None
            except UnicodeEncodeError:
                raise ValueError('sequence should contain ASCII characters only') from None
        except Exception:
            raise ValueError('sequence should be a Seq, MutableSeq, string, or bytes-like object') from None
        n = len(sequence)
        m = self.length
        scores = np.empty(n - m + 1, np.float32)
        logodds = np.array([[self[letter][i] for letter in 'ACGT'] for i in range(m)], float)
        _pwm.calculate(sequence, logodds, scores)
        if len(scores) == 1:
            return scores[0]
        else:
            return scores

    def search(self, sequence, threshold=0.0, both=True, chunksize=10 ** 6):
        """Find hits with PWM score above given threshold.

        A generator function, returning found hits in the given sequence
        with the pwm score higher than the threshold.
        """
        sequence = sequence.upper()
        seq_len = len(sequence)
        motif_l = self.length
        chunk_starts = np.arange(0, seq_len, chunksize)
        if both:
            rc = self.reverse_complement()
        for chunk_start in chunk_starts:
            subseq = sequence[chunk_start:chunk_start + chunksize + motif_l - 1]
            pos_scores = self.calculate(subseq)
            pos_ind = pos_scores >= threshold
            pos_positions = np.where(pos_ind)[0] + chunk_start
            pos_scores = pos_scores[pos_ind]
            if both:
                neg_scores = rc.calculate(subseq)
                neg_ind = neg_scores >= threshold
                neg_positions = np.where(neg_ind)[0] + chunk_start
                neg_scores = neg_scores[neg_ind]
            else:
                neg_positions = np.empty(0, dtype=int)
                neg_scores = np.empty(0, dtype=int)
            chunk_positions = np.append(pos_positions, neg_positions - seq_len)
            chunk_scores = np.append(pos_scores, neg_scores)
            order = np.argsort(np.append(pos_positions, neg_positions))
            chunk_positions = chunk_positions[order]
            chunk_scores = chunk_scores[order]
            yield from zip(chunk_positions, chunk_scores)

    @property
    def max(self):
        """Maximal possible score for this motif.

        returns the score computed for the consensus sequence.
        """
        score = 0.0
        letters = self.alphabet
        for position in range(self.length):
            score += max((self[letter][position] for letter in letters))
        return score

    @property
    def min(self):
        """Minimal possible score for this motif.

        returns the score computed for the anticonsensus sequence.
        """
        score = 0.0
        letters = self.alphabet
        for position in range(self.length):
            score += min((self[letter][position] for letter in letters))
        return score

    @property
    def gc_content(self):
        """Compute the GC-ratio."""
        raise Exception('Cannot compute the %GC composition of a PSSM')

    def mean(self, background=None):
        """Return expected value of the score of a motif."""
        if background is None:
            background = dict.fromkeys(self.alphabet, 1.0)
        else:
            background = dict(background)
        total = sum(background.values())
        for letter in self.alphabet:
            background[letter] /= total
        sx = 0.0
        for i in range(self.length):
            for letter in self.alphabet:
                logodds = self[letter, i]
                if math.isnan(logodds):
                    continue
                if math.isinf(logodds) and logodds < 0:
                    continue
                b = background[letter]
                p = b * math.pow(2, logodds)
                sx += p * logodds
        return sx

    def std(self, background=None):
        """Return standard deviation of the score of a motif."""
        if background is None:
            background = dict.fromkeys(self.alphabet, 1.0)
        else:
            background = dict(background)
        total = sum(background.values())
        for letter in self.alphabet:
            background[letter] /= total
        variance = 0.0
        for i in range(self.length):
            sx = 0.0
            sxx = 0.0
            for letter in self.alphabet:
                logodds = self[letter, i]
                if math.isnan(logodds):
                    continue
                if math.isinf(logodds) and logodds < 0:
                    continue
                b = background[letter]
                p = b * math.pow(2, logodds)
                sx += p * logodds
                sxx += p * logodds * logodds
            sxx -= sx * sx
            variance += sxx
        variance = max(variance, 0)
        return math.sqrt(variance)

    def dist_pearson(self, other):
        """Return the similarity score based on pearson correlation for the given motif against self.

        We use the Pearson's correlation of the respective probabilities.
        """
        if self.alphabet != other.alphabet:
            raise ValueError('Cannot compare motifs with different alphabets')
        max_p = -2
        for offset in range(-self.length + 1, other.length):
            if offset < 0:
                p = self.dist_pearson_at(other, -offset)
            else:
                p = other.dist_pearson_at(self, offset)
            if max_p < p:
                max_p = p
                max_o = -offset
        return (1 - max_p, max_o)

    def dist_pearson_at(self, other, offset):
        """Return the similarity score based on pearson correlation at the given offset."""
        letters = self.alphabet
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        norm = max(self.length, offset + other.length) * len(letters)
        for pos in range(min(self.length - offset, other.length)):
            xi = [self[letter, pos + offset] for letter in letters]
            yi = [other[letter, pos] for letter in letters]
            sx += sum(xi)
            sy += sum(yi)
            sxx += sum((x * x for x in xi))
            sxy += sum((x * y for x, y in zip(xi, yi)))
            syy += sum((y * y for y in yi))
        sx /= norm
        sy /= norm
        sxx /= norm
        sxy /= norm
        syy /= norm
        numerator = sxy - sx * sy
        denominator = math.sqrt((sxx - sx * sx) * (syy - sy * sy))
        return numerator / denominator

    def distribution(self, background=None, precision=10 ** 3):
        """Calculate the distribution of the scores at the given precision."""
        from .thresholds import ScoreDistribution
        if background is None:
            background = dict.fromkeys(self.alphabet, 1.0)
        else:
            background = dict(background)
        total = sum(background.values())
        for letter in self.alphabet:
            background[letter] /= total
        return ScoreDistribution(precision=precision, pssm=self, background=background)