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
class TeacherMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(self, metrics_list: str='default', shared: Dict[str, Any]=None) -> None:
        super().__init__(shared=shared)
        self._metrics_list = self._infer_metrics(metrics_list)
        self.eval_pr = [1, 5, 10, 100]

    @staticmethod
    def _infer_metrics(cli_arg: str) -> Set[str]:
        """
        Parse the CLI metric into a list of metrics we wish to compute.
        """
        col: Set[str] = set()
        names = cli_arg.split(',')
        for n in names:
            if n == 'default':
                col |= DEFAULT_METRICS
            elif n == 'rouge':
                col |= ROUGE_METRICS
            elif n == 'bleu':
                col |= BLEU_METRICS
            elif n == 'all':
                col |= ALL_METRICS
            else:
                col.add(n)
        return col

    def _update_ranking_metrics(self, observation, labels):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            return
        label_set = set((normalize_answer(l) for l in labels))
        cnts = {k: 0 for k in self.eval_pr}
        cnt = 0
        for c in text_cands:
            cnt += 1
            if normalize_answer(c) in label_set:
                for k in self.eval_pr:
                    if cnt <= k:
                        cnts[k] += 1
        for k in self.eval_pr:
            self.add(f'hits@{k}', AverageMetric(cnts[k] > 0))

    def evaluate_response(self, observation: Message, labels: List[str]) -> None:
        """
        Compute all required text-based metrics based on an observation and labels.
        """
        prediction = observation.get('text', None)
        self.add('exs', SumMetric(1))
        if prediction is not None:
            self.add('accuracy', ExactMatchMetric.compute(prediction, labels))
            self.add('f1', F1Metric.compute(prediction, labels))
            for k in range(1, 5):
                if f'bleu-{k}' in self._metrics_list:
                    self.add(f'bleu-{k}', BleuMetric.compute(prediction, labels, k))
            if self._metrics_list & ROUGE_METRICS:
                r1, r2, rL = RougeMetric.compute_many(prediction, labels)
                if 'rouge-1' in self._metrics_list:
                    self.add('rouge_1', r1)
                if 'rouge-2' in self._metrics_list:
                    self.add('rouge_2', r2)
                if 'rouge-L' in self._metrics_list:
                    self.add('rouge_L', rL)
        self._update_ranking_metrics(observation, labels)
        if 'metrics' in observation:
            for uk, v in observation['metrics'].items():
                if uk in ALL_METRICS:
                    uk = f'USER_{uk}'
                assert isinstance(uk, str), type(k)
                if not isinstance(v, Metric):
                    warn_once(f'Metric {uk} is assumed to be averaged per example.')
                    v = AverageMetric(v)
                assert isinstance(v, Metric)
                self.add(uk, v)