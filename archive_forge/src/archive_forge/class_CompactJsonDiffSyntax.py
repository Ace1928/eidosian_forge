import sys
import json
from .symbols import *
from .symbols import Symbol
class CompactJsonDiffSyntax(object):

    def emit_set_diff(self, a, b, s, added, removed):
        if s == 0.0 or len(removed) == len(a):
            return {replace: b} if isinstance(b, dict) else b
        else:
            d = {}
            if removed:
                d[discard] = removed
            if added:
                d[add] = added
            return d

    def emit_list_diff(self, a, b, s, inserted, changed, deleted):
        if s == 0.0:
            return {replace: b} if isinstance(b, dict) else b
        elif s == 1.0:
            return {}
        else:
            d = changed
            if inserted:
                d[insert] = inserted
            if deleted:
                d[delete] = [pos for pos, value in deleted]
            return d

    def emit_dict_diff(self, a, b, s, added, changed, removed):
        if s == 0.0:
            return {replace: b} if isinstance(b, dict) else b
        elif s == 1.0:
            return {}
        else:
            changed.update(added)
            if removed:
                changed[delete] = list(removed.keys())
            return changed

    def emit_value_diff(self, a, b, s):
        if s == 1.0:
            return {}
        else:
            return {replace: b} if isinstance(b, dict) else b

    def patch(self, a, d):
        if isinstance(d, dict):
            if not d:
                return a
            if replace in d:
                return d[replace]
            if isinstance(a, dict):
                a = dict(a)
                for k, v in d.items():
                    if k is delete:
                        for kdel in v:
                            del a[kdel]
                    else:
                        av = a.get(k, missing)
                        if av is missing:
                            a[k] = v
                        else:
                            a[k] = self.patch(av, v)
                return a
            elif isinstance(a, (list, tuple)):
                original_type = type(a)
                a = list(a)
                if delete in d:
                    for pos in d[delete]:
                        a.pop(pos)
                if insert in d:
                    for pos, value in d[insert]:
                        a.insert(pos, value)
                for k, v in d.items():
                    if k is not delete and k is not insert:
                        k = int(k)
                        a[k] = self.patch(a[k], v)
                if original_type is not list:
                    a = original_type(a)
                return a
            elif isinstance(a, set):
                a = set(a)
                if discard in d:
                    for x in d[discard]:
                        a.discard(x)
                if add in d:
                    for x in d[add]:
                        a.add(x)
                return a
        return d