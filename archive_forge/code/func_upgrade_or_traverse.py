from __future__ import annotations
import sys
import eventlet
def upgrade_or_traverse(obj):
    if id(obj) in visited:
        return None
    if isinstance(obj, klass):
        if obj in old_to_new:
            return old_to_new[obj]
        else:
            new = upgrade(obj)
            old_to_new[obj] = new
            return new
    else:
        _upgrade_instances(obj, klass, upgrade, visited, old_to_new)
        return None