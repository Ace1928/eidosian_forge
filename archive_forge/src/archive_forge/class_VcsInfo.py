import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
class VcsInfo:
    name = 'vcs_info'

    def __init__(self, vcs: str, commit_id: str, requested_revision: Optional[str]=None) -> None:
        self.vcs = vcs
        self.requested_revision = requested_revision
        self.commit_id = commit_id

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['VcsInfo']:
        if d is None:
            return None
        return cls(vcs=_get_required(d, str, 'vcs'), commit_id=_get_required(d, str, 'commit_id'), requested_revision=_get(d, str, 'requested_revision'))

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(vcs=self.vcs, requested_revision=self.requested_revision, commit_id=self.commit_id)