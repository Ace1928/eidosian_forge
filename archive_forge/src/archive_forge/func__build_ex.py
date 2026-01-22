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
def _build_ex(self, ex):
    """
        Find the token span of the answer in the context for this example.

        If a token span cannot be found, return None. Otherwise, torchify.
        """
    if 'text' not in ex:
        return
    inputs = {}
    fields = ex['text'].strip().split('\n')
    if len(fields) < 2:
        raise RuntimeError('Invalid input. Is task a QA task?')
    paragraphs, question = (fields[:-1], fields[-1])
    if len(fields) > 2 and self.opt.get('subsample_docs', 0) > 0 and ('labels' in ex):
        paragraphs = self._subsample_doc(paragraphs, ex['labels'], self.opt.get('subsample_docs', 0))
    document = ' '.join(paragraphs)
    inputs['document'], doc_spans = self.word_dict.span_tokenize(document)
    inputs['question'] = self.word_dict.tokenize(question)
    inputs['target'] = None
    if 'labels' in ex:
        if 'answer_starts' in ex:
            labels_with_inds = list(zip(ex['labels'], ex['answer_starts']))
            random.shuffle(labels_with_inds)
            for ans, ch_idx in labels_with_inds:
                start_idx = bisect.bisect_left(list((x[0] for x in doc_spans)), ch_idx)
                end_idx = start_idx + len(self.word_dict.tokenize(ans)) - 1
                if end_idx < len(doc_spans):
                    inputs['target'] = (start_idx, end_idx)
                    break
        else:
            inputs['target'] = self._find_target(inputs['document'], ex['labels'])
        if inputs['target'] is None:
            return
    inputs = vectorize(self.opt, inputs, self.word_dict, self.feature_dict)
    return inputs + (document, doc_spans)