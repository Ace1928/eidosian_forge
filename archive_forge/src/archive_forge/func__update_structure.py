import re
def _update_structure(structure, kvs, path, command):
    from pyrsistent._pmap import pmap
    e = structure.evolver()
    if not path and command is discard:
        for k, v in reversed(kvs):
            discard(e, k)
    else:
        for k, v in kvs:
            is_empty = False
            if v is _EMPTY_SENTINEL:
                if command is discard:
                    continue
                is_empty = True
                v = pmap()
            result = _do_to_path(v, path, command)
            if result is not v or is_empty:
                e[k] = result
    return e.persistent()