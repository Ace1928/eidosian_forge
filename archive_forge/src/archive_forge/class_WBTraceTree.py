import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.data_types
from wandb.sdk.data_types import _dtypes
from wandb.sdk.data_types.base_types.media import Media
class WBTraceTree(Media):
    """Media object for trace tree data.

    Arguments:
        root_span (Span): The root span of the trace tree.
        model_dict (dict, optional): A dictionary containing the model dump.
            NOTE: model_dict is a completely-user-defined dict. The UI will render
            a JSON viewer for this dict, giving special treatment to dictionaries
            with a `_kind` key. This is because model vendors have such different
            serialization formats that we need to be flexible here.
    """
    _log_type = 'wb_trace_tree'

    def __init__(self, root_span: Span, model_dict: typing.Optional[dict]=None):
        super().__init__()
        self._root_span = root_span
        self._model_dict = model_dict

    @classmethod
    def get_media_subdir(cls) -> str:
        return 'media/wb_trace_tree'

    def to_json(self, run: Optional[Union['LocalRun', 'Artifact']]) -> dict:
        res = {'_type': self._log_type}
        if self._model_dict is not None:
            model_dump_str = _safe_serialize(self._model_dict)
            res['model_hash'] = _hash_id(model_dump_str)
            res['model_dict_dumps'] = model_dump_str
        res['root_span_dumps'] = _safe_serialize(dataclasses.asdict(self._root_span))
        return res

    def is_bound(self) -> bool:
        return True