from contextlib import contextmanager
from typing import Dict, List
def _instantiate_rules(self, attr):
    dct = {}
    for base in type(self).mro():
        rules_map = getattr(base, attr, {})
        for type_, rule_classes in rules_map.items():
            new = [rule_cls(self) for rule_cls in rule_classes]
            dct.setdefault(type_, []).extend(new)
    return dct