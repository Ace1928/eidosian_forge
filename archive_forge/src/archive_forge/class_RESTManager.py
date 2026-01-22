import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
class RESTManager:
    """Base class for CRUD operations on objects.

    Derived class must define ``_path`` and ``_obj_cls``.

    ``_path``: Base URL path on which requests will be sent (e.g. '/projects')
    ``_obj_cls``: The class of objects that will be created
    """
    _create_attrs: g_types.RequiredOptional = g_types.RequiredOptional()
    _update_attrs: g_types.RequiredOptional = g_types.RequiredOptional()
    _path: Optional[str] = None
    _obj_cls: Optional[Type[RESTObject]] = None
    _from_parent_attrs: Dict[str, Any] = {}
    _types: Dict[str, Type[g_types.GitlabAttribute]] = {}
    _computed_path: Optional[str]
    _parent: Optional[RESTObject]
    _parent_attrs: Dict[str, Any]
    gitlab: Gitlab

    def __init__(self, gl: Gitlab, parent: Optional[RESTObject]=None) -> None:
        """REST manager constructor.

        Args:
            gl: :class:`~gitlab.Gitlab` connection to use to make requests.
            parent: REST object to which the manager is attached.
        """
        self.gitlab = gl
        self._parent = parent
        self._computed_path = self._compute_path()

    @property
    def parent_attrs(self) -> Optional[Dict[str, Any]]:
        return self._parent_attrs

    def _compute_path(self, path: Optional[str]=None) -> Optional[str]:
        self._parent_attrs = {}
        if path is None:
            path = self._path
        if path is None:
            return None
        if self._parent is None or not self._from_parent_attrs:
            return path
        data: Dict[str, Optional[gitlab.utils.EncodedId]] = {}
        for self_attr, parent_attr in self._from_parent_attrs.items():
            if not hasattr(self._parent, parent_attr):
                data[self_attr] = None
                continue
            data[self_attr] = gitlab.utils.EncodedId(getattr(self._parent, parent_attr))
        self._parent_attrs = data
        return path.format(**data)

    @property
    def path(self) -> Optional[str]:
        return self._computed_path