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
def _kldiv(self, p_counter, q_counter) -> float:
    ptotal = sum(p_counter.values())
    qtotal = sum(q_counter.values())
    kldiv = 0.0
    for word, _ in p_counter.items():
        prob_p = p_counter[word] / ptotal
        prob_q = q_counter[word] / qtotal
        kldiv += prob_p * math.log(1e-20 + prob_q / prob_p)
    return -kldiv