from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set
def _infer_skip_keys(obj: Any, full_schema: dict, processed_refs: Optional[Set[str]]=None) -> List[str]:
    if processed_refs is None:
        processed_refs = set()
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == '$ref':
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                keys.append(v.split('/')[1])
                keys += _infer_skip_keys(ref, full_schema, processed_refs)
            elif isinstance(v, (list, dict)):
                keys += _infer_skip_keys(v, full_schema, processed_refs)
    elif isinstance(obj, list):
        for el in obj:
            keys += _infer_skip_keys(el, full_schema, processed_refs)
    return keys