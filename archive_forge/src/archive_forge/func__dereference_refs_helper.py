from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set
def _dereference_refs_helper(obj: Any, full_schema: Dict[str, Any], skip_keys: Sequence[str], processed_refs: Optional[Set[str]]=None) -> Any:
    if processed_refs is None:
        processed_refs = set()
    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k in skip_keys:
                obj_out[k] = v
            elif k == '$ref':
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                full_ref = _dereference_refs_helper(ref, full_schema, skip_keys, processed_refs)
                processed_refs.remove(v)
                return full_ref
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(v, full_schema, skip_keys, processed_refs)
            else:
                obj_out[k] = v
        return obj_out
    elif isinstance(obj, list):
        return [_dereference_refs_helper(el, full_schema, skip_keys, processed_refs) for el in obj]
    else:
        return obj