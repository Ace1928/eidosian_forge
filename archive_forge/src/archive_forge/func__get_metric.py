from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
@staticmethod
def _get_metric(distance: str, normalize_score: bool=False) -> Callable:
    """
        Get the distance metric function based on the distance type.

        Args:
            distance (str): The distance type.

        Returns:
            Callable: The distance metric function.

        Raises:
            ValueError: If the distance metric is invalid.
        """
    from rapidfuzz import distance as rf_distance
    module_map: Dict[str, Any] = {StringDistance.DAMERAU_LEVENSHTEIN: rf_distance.DamerauLevenshtein, StringDistance.LEVENSHTEIN: rf_distance.Levenshtein, StringDistance.JARO: rf_distance.Jaro, StringDistance.JARO_WINKLER: rf_distance.JaroWinkler, StringDistance.HAMMING: rf_distance.Hamming, StringDistance.INDEL: rf_distance.Indel}
    if distance not in module_map:
        raise ValueError(f'Invalid distance metric: {distance}\nMust be one of: {list(StringDistance)}')
    module = module_map[distance]
    if normalize_score:
        return module.normalized_distance
    else:
        return module.distance