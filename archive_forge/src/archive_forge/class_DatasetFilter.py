import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union
from ..repocard_data import ModelCardData
@dataclass
class DatasetFilter:
    """
    A class that converts human-readable dataset search parameters into ones
    compatible with the REST API. For all parameters capitalization does not
    matter.

    <Tip warning={true}>

        The `DatasetFilter` class is deprecated and will be removed in huggingface_hub>=0.24. Please pass the filter parameters as keyword arguments directly to [`list_datasets`].

    </Tip>

    Args:
        author (`str`, *optional*):
            A string that can be used to identify datasets on
            the Hub by the original uploader (author or organization), such as
            `facebook` or `huggingface`.
        benchmark (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by their official benchmark.
        dataset_name (`str`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by its name, such as `SQAC` or `wikineural`
        language_creators (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub with how the data was curated, such as `crowdsourced` or
            `machine_generated`.
        language (`str` or `List`, *optional*):
            A string or list of strings representing a two-character language to
            filter datasets by on the Hub.
        multilinguality (`str` or `List`, *optional*):
            A string or list of strings representing a filter for datasets that
            contain multiple languages.
        size_categories (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the size of the dataset such as `100K<n<1M` or
            `1M<n<10M`.
        task_categories (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the designed task, such as `audio_classification` or
            `named_entity_recognition`.
        task_ids (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the specific task such as `speech_emotion_recognition` or
            `paraphrase`.

    Examples:

    ```py
    >>> from huggingface_hub import DatasetFilter

    >>> # Using author
    >>> new_filter = DatasetFilter(author="facebook")

    >>> # Using benchmark
    >>> new_filter = DatasetFilter(benchmark="raft")

    >>> # Using dataset_name
    >>> new_filter = DatasetFilter(dataset_name="wikineural")

    >>> # Using language_creator
    >>> new_filter = DatasetFilter(language_creator="crowdsourced")

    >>> # Using language
    >>> new_filter = DatasetFilter(language="en")

    >>> # Using multilinguality
    >>> new_filter = DatasetFilter(multilinguality="multilingual")

    >>> # Using size_categories
    >>> new_filter = DatasetFilter(size_categories="100K<n<1M")

    >>> # Using task_categories
    >>> new_filter = DatasetFilter(task_categories="audio_classification")

    >>> # Using task_ids
    >>> new_filter = DatasetFilter(task_ids="paraphrase")
    ```
    """
    author: Optional[str] = None
    benchmark: Optional[Union[str, List[str]]] = None
    dataset_name: Optional[str] = None
    language_creators: Optional[Union[str, List[str]]] = None
    language: Optional[Union[str, List[str]]] = None
    multilinguality: Optional[Union[str, List[str]]] = None
    size_categories: Optional[Union[str, List[str]]] = None
    task_categories: Optional[Union[str, List[str]]] = None
    task_ids: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        warnings.warn("'DatasetFilter' is deprecated and will be removed in huggingface_hub>=0.24. Please pass the filter parameters as keyword arguments directly to the `list_datasets` method.", category=FutureWarning)