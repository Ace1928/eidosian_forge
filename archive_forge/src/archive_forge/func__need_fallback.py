import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def _need_fallback(self, seek_sequence, seek_outputs, index, logits_processor, generation_config, vocab_size, temperature):
    needs_fallback = False
    should_skip = False
    if generation_config.compression_ratio_threshold is not None:
        compression_ratio = self._retrieve_compression_ratio(seek_sequence, vocab_size)
        if compression_ratio > generation_config.compression_ratio_threshold:
            needs_fallback = True
    if generation_config.logprob_threshold is not None:
        if 'sequences_scores' in seek_outputs[0]:
            logprobs = [s['sequences_scores'] for s in seek_outputs][index]
        else:
            scores = seek_outputs[index]['scores']
            logprobs = self._retrieve_avg_logprobs(scores, seek_sequence, generation_config.eos_token_id, temperature)
        if logprobs < generation_config.logprob_threshold:
            needs_fallback = True
    if generation_config.no_speech_threshold is not None:
        no_speech_prob = _get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, 'no_speech_prob')
        if logprobs < generation_config.logprob_threshold and no_speech_prob[index] > generation_config.no_speech_threshold:
            needs_fallback = False
            should_skip = True
    return (needs_fallback, should_skip)