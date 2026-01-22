import json
import math
import os
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from nltk import ngrams
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import AverageMetric, SumMetric, GlobalAverageMetric
from parlai.utils.misc import round_sigfigs
def _jsdiv(self, dist1: Counter, dist2: Counter) -> float:
    half = dist1 + dist2
    return 0.5 * self._kldiv(dist1, half) + 0.5 * self._kldiv(dist2, half)