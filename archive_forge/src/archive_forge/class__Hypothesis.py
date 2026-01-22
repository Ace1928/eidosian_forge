import warnings
from collections import defaultdict
from math import log
class _Hypothesis:
    """
    Partial solution to a translation.

    Records the word positions of the phrase being translated, its
    translation, raw score, and the cost of the untranslated parts of
    the sentence. When the next phrase is selected to build upon the
    partial solution, a new _Hypothesis object is created, with a back
    pointer to the previous hypothesis.

    To find out which words have been translated so far, look at the
    ``src_phrase_span`` in the hypothesis chain. Similarly, the
    translation output can be found by traversing up the chain.
    """

    def __init__(self, raw_score=0.0, src_phrase_span=(), trg_phrase=(), previous=None, future_score=0.0):
        """
        :param raw_score: Likelihood of hypothesis so far.
            Higher is better. Does not account for untranslated words.
        :type raw_score: float

        :param src_phrase_span: Span of word positions covered by the
            source phrase in this hypothesis expansion. For example,
            (2, 5) means that the phrase is from the second word up to,
            but not including the fifth word in the source sentence.
        :type src_phrase_span: tuple(int)

        :param trg_phrase: Translation of the source phrase in this
            hypothesis expansion
        :type trg_phrase: tuple(str)

        :param previous: Previous hypothesis before expansion to this one
        :type previous: _Hypothesis

        :param future_score: Approximate score for translating the
            remaining words not covered by this hypothesis. Higher means
            that the remaining words are easier to translate.
        :type future_score: float
        """
        self.raw_score = raw_score
        self.src_phrase_span = src_phrase_span
        self.trg_phrase = trg_phrase
        self.previous = previous
        self.future_score = future_score

    def score(self):
        """
        Overall score of hypothesis after accounting for local and
        global features
        """
        return self.raw_score + self.future_score

    def untranslated_spans(self, sentence_length):
        """
        Starting from each untranslated word, find the longest
        continuous span of untranslated positions

        :param sentence_length: Length of source sentence being
            translated by the hypothesis
        :type sentence_length: int

        :rtype: list(tuple(int, int))
        """
        translated_positions = self.translated_positions()
        translated_positions.sort()
        translated_positions.append(sentence_length)
        untranslated_spans = []
        start = 0
        for end in translated_positions:
            if start < end:
                untranslated_spans.append((start, end))
            start = end + 1
        return untranslated_spans

    def translated_positions(self):
        """
        List of positions in the source sentence of words already
        translated. The list is not sorted.

        :rtype: list(int)
        """
        translated_positions = []
        current_hypothesis = self
        while current_hypothesis.previous is not None:
            translated_span = current_hypothesis.src_phrase_span
            translated_positions.extend(range(translated_span[0], translated_span[1]))
            current_hypothesis = current_hypothesis.previous
        return translated_positions

    def total_translated_words(self):
        return len(self.translated_positions())

    def translation_so_far(self):
        translation = []
        self.__build_translation(self, translation)
        return translation

    def __build_translation(self, hypothesis, output):
        if hypothesis.previous is None:
            return
        self.__build_translation(hypothesis.previous, output)
        output.extend(hypothesis.trg_phrase)