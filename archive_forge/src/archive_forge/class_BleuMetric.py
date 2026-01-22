import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
class BleuMetric(AverageMetric):

    @staticmethod
    def compute(guess: str, answers: List[str], k: int=4) -> Optional['BleuMetric']:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        if nltkbleu is None:
            return None
        weights = [1 / k for _ in range(k)]
        score = nltkbleu.sentence_bleu([normalize_answer(a).split(' ') for a in answers], normalize_answer(guess).split(' '), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1, weights=weights)
        return BleuMetric(score)