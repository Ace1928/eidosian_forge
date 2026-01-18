import inspect
from typing import List, Union
import numpy as np
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args

        Classify the sequence(s) given as inputs. See the [`ZeroShotClassificationPipeline`] documentation for more
        information.

        Args:
            sequences (`str` or `List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`. The default template
                works well in many cases, but it may be worthwhile to experiment with different templates depending on
                the task setting.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (`str`) -- The sequence for which this is the output.
            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (`List[float]`) -- The probabilities for each of the labels.
        