from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _chrf_score_update(preds: Union[str, Sequence[str]], target: Union[Sequence[str], Sequence[Sequence[str]]], total_preds_char_n_grams: Dict[int, Tensor], total_preds_word_n_grams: Dict[int, Tensor], total_target_char_n_grams: Dict[int, Tensor], total_target_word_n_grams: Dict[int, Tensor], total_matching_char_n_grams: Dict[int, Tensor], total_matching_word_n_grams: Dict[int, Tensor], n_char_order: int, n_word_order: int, n_order: float, beta: float, lowercase: bool, whitespace: bool, sentence_chrf_score: Optional[List[Tensor]]=None) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Optional[List[Tensor]]]:
    """Update function for chrf score.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        total_preds_char_n_grams: A dictionary containing a total number of hypothesis character n-grams.
        total_preds_word_n_grams: A dictionary containing a total number of hypothesis word n-grams.
        total_target_char_n_grams: A dictionary containing a total number of reference character n-grams.
        total_target_word_n_grams: A dictionary containing a total number of reference word n-grams.
        total_matching_char_n_grams:
            A dictionary containing a total number of matching character n-grams between references and hypotheses.
        total_matching_word_n_grams:
            A dictionary containing a total number of total matching word n-grams between references and hypotheses.
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        n_order: Sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.
        sentence_chrf_score: A list of sentence-level chrF/chrF++ scores.

    Return:
        total_target_char_n_grams: number of reference character n-grams.
        total_target_word_n_grams: number of reference word n-grams.
        total_preds_char_n_grams: number of hypothesis character n-grams.
        total_preds_word_n_grams: number of hypothesis word n-grams.
        total_matching_char_n_grams: number of matching character n-grams between references and hypotheses.
        total_matching_word_n_grams: number of total matching word n-grams between references and hypotheses.
        sentence_chrf_score: A list of sentence-level chrF/chrF++ scores.

    Raises:
        ValueError:
            If length of ``preds`` and ``target`` differs.

    """
    target_corpus, preds = _validate_inputs(target, preds)
    for pred, targets in zip(preds, target_corpus):
        pred_char_n_grams_counts, pred_word_n_grams_counts, pred_char_n_grams, pred_word_n_grams = _get_n_grams_counts_and_total_ngrams(pred, n_char_order, n_word_order, lowercase, whitespace)
        total_preds_char_n_grams = _sum_over_dicts(total_preds_char_n_grams, pred_char_n_grams)
        total_preds_word_n_grams = _sum_over_dicts(total_preds_word_n_grams, pred_word_n_grams)
        sentence_level_f_score, matching_char_n_grams, matching_word_n_grams, target_char_n_grams, target_word_n_grams = _calculate_sentence_level_chrf_score(targets, pred_char_n_grams_counts, pred_word_n_grams_counts, pred_char_n_grams, pred_word_n_grams, n_char_order, n_word_order, n_order, beta, lowercase, whitespace)
        if sentence_chrf_score is not None:
            sentence_chrf_score.append(sentence_level_f_score.unsqueeze(0))
        total_target_char_n_grams = _sum_over_dicts(total_target_char_n_grams, target_char_n_grams)
        total_target_word_n_grams = _sum_over_dicts(total_target_word_n_grams, target_word_n_grams)
        total_matching_char_n_grams = _sum_over_dicts(total_matching_char_n_grams, matching_char_n_grams)
        total_matching_word_n_grams = _sum_over_dicts(total_matching_word_n_grams, matching_word_n_grams)
    return (total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams, sentence_chrf_score)