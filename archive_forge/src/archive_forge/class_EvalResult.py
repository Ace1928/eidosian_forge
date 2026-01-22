import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
@dataclass
class EvalResult:
    """
    Flattened representation of individual evaluation results found in model-index of Model Cards.

    For more information on the model-index spec, see https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1.

    Args:
        task_type (`str`):
            The task identifier. Example: "image-classification".
        dataset_type (`str`):
            The dataset identifier. Example: "common_voice". Use dataset id from https://hf.co/datasets.
        dataset_name (`str`):
            A pretty name for the dataset. Example: "Common Voice (French)".
        metric_type (`str`):
            The metric identifier. Example: "wer". Use metric id from https://hf.co/metrics.
        metric_value (`Any`):
            The metric value. Example: 0.9 or "20.0 ± 1.2".
        task_name (`str`, *optional*):
            A pretty name for the task. Example: "Speech Recognition".
        dataset_config (`str`, *optional*):
            The name of the dataset configuration used in `load_dataset()`.
            Example: fr in `load_dataset("common_voice", "fr")`. See the `datasets` docs for more info:
            https://hf.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name
        dataset_split (`str`, *optional*):
            The split used in `load_dataset()`. Example: "test".
        dataset_revision (`str`, *optional*):
            The revision (AKA Git Sha) of the dataset used in `load_dataset()`.
            Example: 5503434ddd753f426f4b38109466949a1217c2bb
        dataset_args (`Dict[str, Any]`, *optional*):
            The arguments passed during `Metric.compute()`. Example for `bleu`: `{"max_order": 4}`
        metric_name (`str`, *optional*):
            A pretty name for the metric. Example: "Test WER".
        metric_config (`str`, *optional*):
            The name of the metric configuration used in `load_metric()`.
            Example: bleurt-large-512 in `load_metric("bleurt", "bleurt-large-512")`.
            See the `datasets` docs for more info: https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations
        metric_args (`Dict[str, Any]`, *optional*):
            The arguments passed during `Metric.compute()`. Example for `bleu`: max_order: 4
        verified (`bool`, *optional*):
            Indicates whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not. Automatically computed by Hugging Face, do not set.
        verify_token (`str`, *optional*):
            A JSON Web Token that is used to verify whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not.
        source_name (`str`, *optional*):
            The name of the source of the evaluation result. Example: "Open LLM Leaderboard".
        source_url (`str`, *optional*):
            The URL of the source of the evaluation result. Example: "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard".
    """
    task_type: str
    dataset_type: str
    dataset_name: str
    metric_type: str
    metric_value: Any
    task_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_revision: Optional[str] = None
    dataset_args: Optional[Dict[str, Any]] = None
    metric_name: Optional[str] = None
    metric_config: Optional[str] = None
    metric_args: Optional[Dict[str, Any]] = None
    verified: Optional[bool] = None
    verify_token: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None

    @property
    def unique_identifier(self) -> tuple:
        """Returns a tuple that uniquely identifies this evaluation."""
        return (self.task_type, self.dataset_type, self.dataset_config, self.dataset_split, self.dataset_revision)

    def is_equal_except_value(self, other: 'EvalResult') -> bool:
        """
        Return True if `self` and `other` describe exactly the same metric but with a
        different value.
        """
        for key, _ in self.__dict__.items():
            if key == 'metric_value':
                continue
            if key != 'verify_token' and getattr(self, key) != getattr(other, key):
                return False
        return True

    def __post_init__(self) -> None:
        if self.source_name is not None and self.source_url is None:
            raise ValueError('If `source_name` is provided, `source_url` must also be provided.')