from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
Calculate `BLEU score`_ of machine translated text with one or more references.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing - see [2]
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Return:
        Tensor with BLEU Score

    Raises:
        ValueError: If ``preds`` and ``target`` corpus have different lengths.
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.functional.text import bleu_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> bleu_score(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_

    