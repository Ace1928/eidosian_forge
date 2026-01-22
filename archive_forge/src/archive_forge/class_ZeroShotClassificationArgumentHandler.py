import inspect
from typing import List, Union
import numpy as np
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(',') if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError('You must include at least one label and at least one sequence.')
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError('The provided hypothesis_template "{}" was not able to be formatted with the target labels. Make sure the passed template includes formatting syntax such as {{}} where the label should go.'.format(hypothesis_template))
        if isinstance(sequences, str):
            sequences = [sequences]
        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])
        return (sequence_pairs, sequences)