import builtins
import os
import sys
import types
def whats_changed(obj, seen=None, simple=False, first=True):
    """
    Check an object against the memo. Returns a list in the form
    (attribute changes, container changed). Attribute changes is a dict of
    attribute name to attribute value. container changed is a boolean.
    If simple is true, just returns a boolean. None for either item means
    that it has not been checked yet
    """
    if first:
        if '_' in builtins.__dict__:
            del builtins._
        if seen is None:
            seen = {}
    obj_id = id(obj)
    if obj_id in seen:
        if simple:
            return any(seen[obj_id])
        return seen[obj_id]
    if obj_id in dont_memo:
        seen[obj_id] = [{}, False]
        if simple:
            return False
        return seen[obj_id]
    elif obj_id not in memo:
        if simple:
            return True
        else:
            raise RuntimeError('Object not memorised ' + str(obj))
    seen[obj_id] = ({}, False)
    chngd = whats_changed
    id_ = id
    attrs = get_attrs(obj)
    if attrs is None:
        changed = {}
    else:
        obj_attrs = memo[obj_id][0]
        obj_get = obj_attrs.get
        changed = dict(((key, None) for key in obj_attrs if key not in attrs))
        for key, o in attrs.items():
            if id_(o) != obj_get(key, None) or chngd(o, seen, True, False):
                changed[key] = o
    items = get_seq(obj)
    seq_diff = False
    if items is not None and hasattr(items, '__len__'):
        obj_seq = memo[obj_id][1]
        if len(items) != len(obj_seq):
            seq_diff = True
        elif hasattr(obj, 'items'):
            obj_get = obj_seq.get
            for key, item in items.items():
                if id_(item) != obj_get(id_(key)) or chngd(key, seen, True, False) or chngd(item, seen, True, False):
                    seq_diff = True
                    break
        else:
            for i, j in zip(items, obj_seq):
                if id_(i) != j or chngd(i, seen, True, False):
                    seq_diff = True
                    break
    seen[obj_id] = (changed, seq_diff)
    if simple:
        return changed or seq_diff
    return (changed, seq_diff)