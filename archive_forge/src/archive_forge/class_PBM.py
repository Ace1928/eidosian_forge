import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
class PBM:
    """Simulate click data with position bias model. There are other models available in
    `ULTRA <https://github.com/ULTR-Community/ULTRA.git>`_ like the cascading model.

    References
    ----------
    Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm

    """

    def __init__(self, eta: float) -> None:
        self.click_prob = np.array([0.1, 0.16, 0.28, 0.52, 1.0])
        exam_prob = np.array([0.68, 0.61, 0.48, 0.34, 0.28, 0.2, 0.11, 0.1, 0.08, 0.06])
        self.exam_prob = np.power(exam_prob, eta)

    def sample_clicks_for_query(self, labels: npt.NDArray[np.int32], position: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
        """Sample clicks for one query based on input relevance degree and position.

        Parameters
        ----------

        labels :
            relevance_degree

        """
        labels = np.array(labels, copy=True)
        click_prob = np.zeros(labels.shape)
        labels[labels < 0] = 0
        labels[labels >= len(self.click_prob)] = -1
        click_prob = self.click_prob[labels]
        exam_prob = np.zeros(labels.shape)
        assert position.size == labels.size
        ranks = np.array(position, copy=True)
        ranks[ranks >= self.exam_prob.size] = -1
        exam_prob = self.exam_prob[ranks]
        rng = np.random.default_rng(1994)
        prob = rng.random(size=labels.shape[0], dtype=np.float32)
        clicks: npt.NDArray[np.int32] = np.zeros(labels.shape, dtype=np.int32)
        clicks[prob < exam_prob * click_prob] = 1
        return clicks