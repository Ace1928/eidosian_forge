import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
class ScoreConfig:
    """
        Config to present human-friendly evaluation results.
    """

    def __init__(self, score_type=ScoreType.Rel, multiplier=100, score_level=0.01, interval_level=0.01, overfit_iterations_info=True):
        """

        :param score_type: type of score. For abs difference score will be (baseline - test).mean(),
        for relative it's ((baseline - test) / baseline).mean()
        :param multiplier: multiplier to print score
        :param score_level: WX-test level. Will be used to make if tested case significantly better or worse
        :param interval_level: level to compute score confidence interval
        :param overfit_iterations_info: if information about overfit iterations should be preserved
        """
        self.type = score_type
        self.multiplier = multiplier
        self.score_level = score_level
        self.interval_level = interval_level
        self.overfit_overfit_iterations_info = overfit_iterations_info

    @staticmethod
    def abs_score(level=0.01):
        return ScoreConfig(score_type=ScoreType.Abs, multiplier=1, score_level=level)

    @staticmethod
    def rel_score(level=0.01):
        return ScoreConfig(score_type=ScoreType.Rel, multiplier=100, score_level=level)