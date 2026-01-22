import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.first_call = True

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings."""
        if self.first_call:
            self.generated_tokens = [1 for _ in range(input_ids.shape[0])]
            self.start_length = input_ids.shape[-1] - 1
            self.first_call = False
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for i, decoded_generation in enumerate(decoded_generations):
            sequence_complete = any((stop_string in decoded_generation for stop_string in self.stop_strings))
            done.append(sequence_complete)
            if not sequence_complete:
                self.generated_tokens[i] += 1
        if all(done):
            self.first_call = True
        return all(done)