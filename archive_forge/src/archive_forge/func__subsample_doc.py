import bisect
import os
import numpy as np
import json
import random
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.build_data import modelzoo_path
from . import config
from .utils import build_feature_dict, vectorize, batchify, normalize_text
from .model import DocReaderModel
def _subsample_doc(self, paras, labels, subsample):
    """
        Subsample paragraphs from the document (mostly for training speed).
        """
    pi = -1
    for ind, p in enumerate(paras):
        for l in labels:
            if p.find(l):
                pi = ind
                break
    if pi == -1:
        return paras[0:1]
    new_paras = []
    if pi > 0:
        for _i in range(min(subsample, pi - 1)):
            ind = random.randint(0, pi - 1)
            new_paras.append(paras[ind])
    new_paras.append(paras[pi])
    if pi < len(paras) - 1:
        for _i in range(min(subsample, len(paras) - 1 - pi)):
            ind = random.randint(pi + 1, len(paras) - 1)
            new_paras.append(paras[ind])
    return new_paras