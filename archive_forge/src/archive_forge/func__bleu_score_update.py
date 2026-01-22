from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
def _bleu_score_update(preds: Sequence[str], target: Sequence[Sequence[str]], numerator: Tensor, denominator: Tensor, preds_len: Tensor, target_len: Tensor, n_gram: int=4, tokenizer: Callable[[str], Sequence[str]]=_tokenize_fn) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute the BLEU score.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        numerator: Numerator of precision score (true positives)
        denominator: Denominator of precision score (true positives + false positives)
        preds_len: count of words in a candidate prediction
        target_len: count of words in a reference translation
        target: count of words in a reference translation
        n_gram: gram value ranged 1 to 4
        tokenizer: A function that turns sentence into list of words

    """
    target_: Sequence[Sequence[Sequence[str]]] = [[tokenizer(line) if line else [] for line in t] for t in target]
    preds_: Sequence[Sequence[str]] = [tokenizer(line) if line else [] for line in preds]
    for pred, targets in zip(preds_, target_):
        preds_len += len(pred)
        target_len_list = [len(tgt) for tgt in targets]
        target_len_diff = [abs(len(pred) - x) for x in target_len_list]
        target_len += target_len_list[target_len_diff.index(min(target_len_diff))]
        preds_counter: Counter = _count_ngram(pred, n_gram)
        target_counter: Counter = Counter()
        for tgt in targets:
            target_counter |= _count_ngram(tgt, n_gram)
        ngram_counter_clip = preds_counter & target_counter
        for counter_clip in ngram_counter_clip:
            numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]
        for counter in preds_counter:
            denominator[len(counter) - 1] += preds_counter[counter]
    return (preds_len, target_len)