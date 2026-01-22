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
def _l2dist(self, bins):
    return (bins.get('frequent', 0) - 0.4) ** 2 + (bins.get('medium', 0) - 0.3) ** 2 + (bins.get('rare', 0) - 0.2) ** 2 + (bins.get('veryrare', 0) - 0.1) ** 2 + (bins.get('never', 0) - 0.0) ** 2