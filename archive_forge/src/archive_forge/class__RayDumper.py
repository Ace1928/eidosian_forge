import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
class _RayDumper(yaml.SafeDumper):

    def represent_sequence(self, tag, sequence, flow_style=None):
        if len(sequence) > _SEQUENCE_LEN_FLOW_STYLE:
            return super().represent_sequence(tag, sequence, flow_style=True)
        return super().represent_sequence(tag, sequence, flow_style=flow_style)