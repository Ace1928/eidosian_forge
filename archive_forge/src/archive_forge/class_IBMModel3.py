import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel2
from nltk.translate.ibm_model import Counts
class IBMModel3(IBMModel):
    """
    Translation model that considers how a word can be aligned to
    multiple words in another language

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'war', 'ja', 'groß'], ['the', 'house', 'was', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['ein', 'haus', 'ist', 'klein'], ['a', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))
    >>> bitext.append(AlignedSent(['ich', 'fasse', 'das', 'buch', 'zusammen'], ['i', 'summarize', 'the', 'book']))
    >>> bitext.append(AlignedSent(['fasse', 'zusammen'], ['summarize']))

    >>> ibm3 = IBMModel3(bitext, 5)

    >>> print(round(ibm3.translation_table['buch']['book'], 3))
    1.0
    >>> print(round(ibm3.translation_table['das']['book'], 3))
    0.0
    >>> print(round(ibm3.translation_table['ja'][None], 3))
    1.0

    >>> print(round(ibm3.distortion_table[1][1][2][2], 3))
    1.0
    >>> print(round(ibm3.distortion_table[1][2][2][2], 3))
    0.0
    >>> print(round(ibm3.distortion_table[2][2][4][5], 3))
    0.75

    >>> print(round(ibm3.fertility_table[2]['summarize'], 3))
    1.0
    >>> print(round(ibm3.fertility_table[1]['book'], 3))
    1.0

    >>> print(round(ibm3.p1, 3))
    0.054

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, None), (4, 3)])

    """

    def __init__(self, sentence_aligned_corpus, iterations, probability_tables=None):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model, a distortion model, a fertility model, and a
        model for generating NULL-aligned words.

        Translation direction is from ``AlignedSent.mots`` to
        ``AlignedSent.words``.

        :param sentence_aligned_corpus: Sentence-aligned parallel corpus
        :type sentence_aligned_corpus: list(AlignedSent)

        :param iterations: Number of iterations to run training algorithm
        :type iterations: int

        :param probability_tables: Optional. Use this to pass in custom
            probability values. If not specified, probabilities will be
            set to a uniform distribution, or some other sensible value.
            If specified, all the following entries must be present:
            ``translation_table``, ``alignment_table``,
            ``fertility_table``, ``p1``, ``distortion_table``.
            See ``IBMModel`` for the type and purpose of these tables.
        :type probability_tables: dict[str]: object
        """
        super().__init__(sentence_aligned_corpus)
        self.reset_probabilities()
        if probability_tables is None:
            ibm2 = IBMModel2(sentence_aligned_corpus, iterations)
            self.translation_table = ibm2.translation_table
            self.alignment_table = ibm2.alignment_table
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            self.translation_table = probability_tables['translation_table']
            self.alignment_table = probability_tables['alignment_table']
            self.fertility_table = probability_tables['fertility_table']
            self.p1 = probability_tables['p1']
            self.distortion_table = probability_tables['distortion_table']
        for n in range(0, iterations):
            self.train(sentence_aligned_corpus)

    def reset_probabilities(self):
        super().reset_probabilities()
        self.distortion_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))))
        '\n        dict[int][int][int][int]: float. Probability(j | i,l,m).\n        Values accessed as ``distortion_table[j][i][l][m]``.\n        '

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        l_m_combinations = set()
        for aligned_sentence in sentence_aligned_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            if (l, m) not in l_m_combinations:
                l_m_combinations.add((l, m))
                initial_prob = 1 / m
                if initial_prob < IBMModel.MIN_PROB:
                    warnings.warn('A target sentence is too long (' + str(m) + ' words). Results may be less accurate.')
                for j in range(1, m + 1):
                    for i in range(0, l + 1):
                        self.distortion_table[j][i][l][m] = initial_prob
        self.fertility_table[0] = defaultdict(lambda: 0.2)
        self.fertility_table[1] = defaultdict(lambda: 0.65)
        self.fertility_table[2] = defaultdict(lambda: 0.1)
        self.fertility_table[3] = defaultdict(lambda: 0.04)
        MAX_FERTILITY = 10
        initial_fert_prob = 0.01 / (MAX_FERTILITY - 4)
        for phi in range(4, MAX_FERTILITY):
            self.fertility_table[phi] = defaultdict(lambda: initial_fert_prob)
        self.p1 = 0.5

    def train(self, parallel_corpus):
        counts = Model3Counts()
        for aligned_sentence in parallel_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            sampled_alignments, best_alignment = self.sample(aligned_sentence)
            aligned_sentence.alignment = Alignment(best_alignment.zero_indexed_alignment())
            total_count = self.prob_of_alignments(sampled_alignments)
            for alignment_info in sampled_alignments:
                count = self.prob_t_a_given_s(alignment_info)
                normalized_count = count / total_count
                for j in range(1, m + 1):
                    counts.update_lexical_translation(normalized_count, alignment_info, j)
                    counts.update_distortion(normalized_count, alignment_info, j, l, m)
                counts.update_null_generation(normalized_count, alignment_info)
                counts.update_fertility(normalized_count, alignment_info)
        existing_alignment_table = self.alignment_table
        self.reset_probabilities()
        self.alignment_table = existing_alignment_table
        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_distortion_probabilities(counts)
        self.maximize_fertility_probabilities(counts)
        self.maximize_null_generation_probabilities(counts)

    def maximize_distortion_probabilities(self, counts):
        MIN_PROB = IBMModel.MIN_PROB
        for j, i_s in counts.distortion.items():
            for i, src_sentence_lengths in i_s.items():
                for l, trg_sentence_lengths in src_sentence_lengths.items():
                    for m in trg_sentence_lengths:
                        estimate = counts.distortion[j][i][l][m] / counts.distortion_for_any_j[i][l][m]
                        self.distortion_table[j][i][l][m] = max(estimate, MIN_PROB)

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        src_sentence = alignment_info.src_sentence
        trg_sentence = alignment_info.trg_sentence
        l = len(src_sentence) - 1
        m = len(trg_sentence) - 1
        p1 = self.p1
        p0 = 1 - p1
        probability = 1.0
        MIN_PROB = IBMModel.MIN_PROB
        null_fertility = alignment_info.fertility_of_i(0)
        probability *= pow(p1, null_fertility) * pow(p0, m - 2 * null_fertility)
        if probability < MIN_PROB:
            return MIN_PROB
        for i in range(1, null_fertility + 1):
            probability *= (m - null_fertility - i + 1) / i
            if probability < MIN_PROB:
                return MIN_PROB
        for i in range(1, l + 1):
            fertility = alignment_info.fertility_of_i(i)
            probability *= factorial(fertility) * self.fertility_table[fertility][src_sentence[i]]
            if probability < MIN_PROB:
                return MIN_PROB
        for j in range(1, m + 1):
            t = trg_sentence[j]
            i = alignment_info.alignment[j]
            s = src_sentence[i]
            probability *= self.translation_table[t][s] * self.distortion_table[j][i][l][m]
            if probability < MIN_PROB:
                return MIN_PROB
        return probability