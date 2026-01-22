import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktSentenceTokenizer(PunktBaseClass, TokenizerI):
    """
    A sentence tokenizer which uses an unsupervised algorithm to build
    a model for abbreviation words, collocations, and words that start
    sentences; and then uses that model to find sentence boundaries.
    This approach has been shown to work well for many European
    languages.
    """

    def __init__(self, train_text=None, verbose=False, lang_vars=None, token_cls=PunktToken):
        """
        train_text can either be the sole training text for this sentence
        boundary detector, or can be a PunktParameters object.
        """
        PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)
        if train_text:
            self._params = self.train(train_text, verbose)

    def train(self, train_text, verbose=False):
        """
        Derives parameters from a given training text, or uses the parameters
        given. Repeated calls to this method destroy previous parameters. For
        incremental training, instantiate a separate PunktTrainer instance.
        """
        if not isinstance(train_text, str):
            return train_text
        return PunktTrainer(train_text, lang_vars=self._lang_vars, token_cls=self._Token).get_params()

    def tokenize(self, text: str, realign_boundaries: bool=True) -> List[str]:
        """
        Given a text, returns a list of the sentences in that text.
        """
        return list(self.sentences_from_text(text, realign_boundaries))

    def debug_decisions(self, text: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies candidate periods as sentence breaks, yielding a dict for
        each that may be used to understand why the decision was made.

        See format_debug_decision() to help make this output readable.
        """
        for match, decision_text in self._match_potential_end_contexts(text):
            tokens = self._tokenize_words(decision_text)
            tokens = list(self._annotate_first_pass(tokens))
            while tokens and (not tokens[0].tok.endswith(self._lang_vars.sent_end_chars)):
                tokens.pop(0)
            yield {'period_index': match.end() - 1, 'text': decision_text, 'type1': tokens[0].type, 'type2': tokens[1].type, 'type1_in_abbrs': bool(tokens[0].abbr), 'type1_is_initial': bool(tokens[0].is_initial), 'type2_is_sent_starter': tokens[1].type_no_sentperiod in self._params.sent_starters, 'type2_ortho_heuristic': self._ortho_heuristic(tokens[1]), 'type2_ortho_contexts': set(self._params._debug_ortho_context(tokens[1].type_no_sentperiod)), 'collocation': (tokens[0].type_no_sentperiod, tokens[1].type_no_sentperiod) in self._params.collocations, 'reason': self._second_pass_annotation(tokens[0], tokens[1]) or REASON_DEFAULT_DECISION, 'break_decision': tokens[0].sentbreak}

    def span_tokenize(self, text: str, realign_boundaries: bool=True) -> Iterator[Tuple[int, int]]:
        """
        Given a text, generates (start, end) spans of sentences
        in the text.
        """
        slices = self._slices_from_text(text)
        if realign_boundaries:
            slices = self._realign_boundaries(text, slices)
        for sentence in slices:
            yield (sentence.start, sentence.stop)

    def sentences_from_text(self, text: str, realign_boundaries: bool=True) -> List[str]:
        """
        Given a text, generates the sentences in that text by only
        testing candidate sentence breaks. If realign_boundaries is
        True, includes in the sentence closing punctuation that
        follows the period.
        """
        return [text[s:e] for s, e in self.span_tokenize(text, realign_boundaries)]

    def _get_last_whitespace_index(self, text: str) -> int:
        """
        Given a text, find the index of the *last* occurrence of *any*
        whitespace character, i.e. " ", "
", "	", "\r", etc.
        If none is found, return 0.
        """
        for i in range(len(text) - 1, -1, -1):
            if text[i] in string.whitespace:
                return i
        return 0

    def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[Match, str]]:
        """
        Given a text, find the matches of potential sentence breaks,
        alongside the contexts surrounding these sentence breaks.

        Since the fix for the ReDOS discovered in issue #2866, we no longer match
        the word before a potential end of sentence token. Instead, we use a separate
        regex for this. As a consequence, `finditer`'s desire to find non-overlapping
        matches no longer aids us in finding the single longest match.
        Where previously, we could use::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +SKIP
            [<re.Match object; span=(9, 18), match='acting!!!'>]

        Now we have to find the word before (i.e. 'acting') separately, and `finditer`
        returns::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._lang_vars.period_context_re().finditer(text)) # doctest: +NORMALIZE_WHITESPACE
            [<re.Match object; span=(15, 16), match='!'>,
            <re.Match object; span=(16, 17), match='!'>,
            <re.Match object; span=(17, 18), match='!'>]

        So, we need to find the word before the match from right to left, and then manually remove
        the overlaps. That is what this method does::

            >>> pst = PunktSentenceTokenizer()
            >>> text = "Very bad acting!!! I promise."
            >>> list(pst._match_potential_end_contexts(text))
            [(<re.Match object; span=(17, 18), match='!'>, 'acting!!! I')]

        :param text: String of one or more sentences
        :type text: str
        :return: Generator of match-context tuples.
        :rtype: Iterator[Tuple[Match, str]]
        """
        previous_slice = slice(0, 0)
        previous_match = None
        for match in self._lang_vars.period_context_re().finditer(text):
            before_text = text[previous_slice.stop:match.start()]
            index_after_last_space = self._get_last_whitespace_index(before_text)
            if index_after_last_space:
                index_after_last_space += previous_slice.stop + 1
            else:
                index_after_last_space = previous_slice.start
            prev_word_slice = slice(index_after_last_space, match.start())
            if previous_match and previous_slice.stop <= prev_word_slice.start:
                yield (previous_match, text[previous_slice] + previous_match.group() + previous_match.group('after_tok'))
            previous_match = match
            previous_slice = prev_word_slice
        if previous_match:
            yield (previous_match, text[previous_slice] + previous_match.group() + previous_match.group('after_tok'))

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        last_break = 0
        for match, context in self._match_potential_end_contexts(text):
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                if match.group('next_tok'):
                    last_break = match.start('next_tok')
                else:
                    last_break = match.end()
        yield slice(last_break, len(text.rstrip()))

    def _realign_boundaries(self, text: str, slices: Iterator[slice]) -> Iterator[slice]:
        """
        Attempts to realign punctuation that falls after the period but
        should otherwise be included in the same sentence.

        For example: "(Sent1.) Sent2." will otherwise be split as::

            ["(Sent1.", ") Sent1."].

        This method will produce::

            ["(Sent1.)", "Sent2."].
        """
        realign = 0
        for sentence1, sentence2 in _pair_iter(slices):
            sentence1 = slice(sentence1.start + realign, sentence1.stop)
            if not sentence2:
                if text[sentence1]:
                    yield sentence1
                continue
            m = self._lang_vars.re_boundary_realignment.match(text[sentence2])
            if m:
                yield slice(sentence1.start, sentence2.start + len(m.group(0).rstrip()))
                realign = m.end()
            else:
                realign = 0
                if text[sentence1]:
                    yield sentence1

    def text_contains_sentbreak(self, text: str) -> bool:
        """
        Returns True if the given text includes a sentence break.
        """
        found = False
        for tok in self._annotate_tokens(self._tokenize_words(text)):
            if found:
                return True
            if tok.sentbreak:
                found = True
        return False

    def sentences_from_text_legacy(self, text: str) -> Iterator[str]:
        """
        Given a text, generates the sentences in that text. Annotates all
        tokens, rather than just those with possible sentence breaks. Should
        produce the same results as ``sentences_from_text``.
        """
        tokens = self._annotate_tokens(self._tokenize_words(text))
        return self._build_sentence_list(text, tokens)

    def sentences_from_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Given a sequence of tokens, generates lists of tokens, each list
        corresponding to a sentence.
        """
        tokens = iter(self._annotate_tokens((self._Token(t) for t in tokens)))
        sentence = []
        for aug_tok in tokens:
            sentence.append(aug_tok.tok)
            if aug_tok.sentbreak:
                yield sentence
                sentence = []
        if sentence:
            yield sentence

    def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Given a set of tokens augmented with markers for line-start and
        paragraph-start, returns an iterator through those tokens with full
        annotation including predicted sentence breaks.
        """
        tokens = self._annotate_first_pass(tokens)
        tokens = self._annotate_second_pass(tokens)
        return tokens

    def _build_sentence_list(self, text: str, tokens: Iterator[PunktToken]) -> Iterator[str]:
        """
        Given the original text and the list of augmented word tokens,
        construct and return a tokenized list of sentence strings.
        """
        pos = 0
        white_space_regexp = re.compile('\\s*')
        sentence = ''
        for aug_tok in tokens:
            tok = aug_tok.tok
            white_space = white_space_regexp.match(text, pos).group()
            pos += len(white_space)
            if text[pos:pos + len(tok)] != tok:
                pat = '\\s*'.join((re.escape(c) for c in tok))
                m = re.compile(pat).match(text, pos)
                if m:
                    tok = m.group()
            assert text[pos:pos + len(tok)] == tok
            pos += len(tok)
            if sentence:
                sentence += white_space
            sentence += tok
            if aug_tok.sentbreak:
                yield sentence
                sentence = ''
        if sentence:
            yield sentence

    def dump(self, tokens: Iterator[PunktToken]) -> None:
        print('writing to /tmp/punkt.new...')
        with open('/tmp/punkt.new', 'w') as outfile:
            for aug_tok in tokens:
                if aug_tok.parastart:
                    outfile.write('\n\n')
                elif aug_tok.linestart:
                    outfile.write('\n')
                else:
                    outfile.write(' ')
                outfile.write(str(aug_tok))
    PUNCTUATION = tuple(';:,.!?')

    def _annotate_second_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Performs a token-based classification (section 4) over the given
        tokens, making use of the orthographic heuristic (4.1.1), collocation
        heuristic (4.1.2) and frequent sentence starter heuristic (4.1.3).
        """
        for token1, token2 in _pair_iter(tokens):
            self._second_pass_annotation(token1, token2)
            yield token1

    def _second_pass_annotation(self, aug_tok1: PunktToken, aug_tok2: Optional[PunktToken]) -> Optional[str]:
        """
        Performs token-based classification over a pair of contiguous tokens
        updating the first.
        """
        if not aug_tok2:
            return
        if not aug_tok1.period_final:
            return
        typ = aug_tok1.type_no_period
        next_typ = aug_tok2.type_no_sentperiod
        tok_is_initial = aug_tok1.is_initial
        if (typ, next_typ) in self._params.collocations:
            aug_tok1.sentbreak = False
            aug_tok1.abbr = True
            return REASON_KNOWN_COLLOCATION
        if (aug_tok1.abbr or aug_tok1.ellipsis) and (not tok_is_initial):
            is_sent_starter = self._ortho_heuristic(aug_tok2)
            if is_sent_starter == True:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC
            if aug_tok2.first_upper and next_typ in self._params.sent_starters:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_SENTENCE_STARTER
        if tok_is_initial or typ == '##number##':
            is_sent_starter = self._ortho_heuristic(aug_tok2)
            if is_sent_starter == False:
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                if tok_is_initial:
                    return REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC
                return REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC
            if is_sent_starter == 'unknown' and tok_is_initial and aug_tok2.first_upper and (not self._params.ortho_context[next_typ] & _ORTHO_LC):
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                return REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC
        return

    def _ortho_heuristic(self, aug_tok: PunktToken) -> Union[bool, str]:
        """
        Decide whether the given token is the first token in a sentence.
        """
        if aug_tok.tok in self.PUNCTUATION:
            return False
        ortho_context = self._params.ortho_context[aug_tok.type_no_sentperiod]
        if aug_tok.first_upper and ortho_context & _ORTHO_LC and (not ortho_context & _ORTHO_MID_UC):
            return True
        if aug_tok.first_lower and (ortho_context & _ORTHO_UC or not ortho_context & _ORTHO_BEG_LC):
            return False
        return 'unknown'