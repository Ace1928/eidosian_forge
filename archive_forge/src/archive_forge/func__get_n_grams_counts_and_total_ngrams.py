from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _get_n_grams_counts_and_total_ngrams(sentence: str, n_char_order: int, n_word_order: int, lowercase: bool, whitespace: bool) -> Tuple[Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Tensor], Dict[int, Tensor]]:
    """Get n-grams and total n-grams.

    Args:
        sentence: An input sentence
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.

    Return:
        char_n_grams_counts: A dictionary of dictionaries with sentence character n-grams.
        word_n_grams_counts: A dictionary of dictionaries with sentence word n-grams.
        total_char_n_grams: A dictionary containing a total number of sentence character n-grams.
        total_word_n_grams: A dictionary containing a total number of sentence word n-grams.

    """

    def _char_and_word_ngrams_counts(sentence: str, n_char_order: int, n_word_order: int, lowercase: bool) -> Tuple[Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Dict[Tuple[str, ...], Tensor]]]:
        """Get a dictionary of dictionaries with a counts of given n-grams."""
        if lowercase:
            sentence = sentence.lower()
        char_n_grams_counts = _ngram_counts(_get_characters(sentence, whitespace), n_char_order)
        word_n_grams_counts = _ngram_counts(_get_words_and_punctuation(sentence), n_word_order)
        return (char_n_grams_counts, word_n_grams_counts)

    def _get_total_ngrams(n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]]) -> Dict[int, Tensor]:
        """Get total sum of n-grams over n-grams w.r.t n."""
        total_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
        for n in n_grams_counts:
            total_n_grams[n] = tensor(sum(n_grams_counts[n].values()))
        return total_n_grams
    char_n_grams_counts, word_n_grams_counts = _char_and_word_ngrams_counts(sentence, n_char_order, n_word_order, lowercase)
    total_char_n_grams = _get_total_ngrams(char_n_grams_counts)
    total_word_n_grams = _get_total_ngrams(word_n_grams_counts)
    return (char_n_grams_counts, word_n_grams_counts, total_char_n_grams, total_word_n_grams)