import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _rouge_score_compute(sentence_results: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
    """Compute the combined ROUGE metric for all the input set of predicted and target sentences.

    Args:
        sentence_results: Rouge-N/Rouge-L/Rouge-LSum metrics calculated for single sentence.

    """
    results: Dict[str, Tensor] = {}
    if sentence_results == {}:
        return results
    for rouge_key, scores in sentence_results.items():
        results[rouge_key] = torch.tensor(scores).mean()
    return results