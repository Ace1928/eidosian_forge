from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from streamlit.runtime.secrets import AttrDict, secrets_singleton
from streamlit.util import calc_md5
@property
def _secrets(self) -> AttrDict:
    """Get the secrets for this connection from the corresponding st.secrets section.

        We expect this property to be used primarily by connection authors when they
        are implementing their class' ``_connect`` method. User scripts should, for the
        most part, have no reason to use this property.
        """
    connections_section = None
    if secrets_singleton.load_if_toml_exists():
        connections_section = secrets_singleton.get('connections')
    if type(connections_section) is not AttrDict:
        return AttrDict({})
    return connections_section.get(self._connection_name, AttrDict({}))