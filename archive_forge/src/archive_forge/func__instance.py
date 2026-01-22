from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from streamlit.runtime.secrets import AttrDict, secrets_singleton
from streamlit.util import calc_md5
@property
def _instance(self) -> RawConnectionT:
    """Get an instance of the underlying connection, creating a new one if needed."""
    if self._raw_instance is None:
        self._raw_instance = self._connect(**self._kwargs)
    return self._raw_instance