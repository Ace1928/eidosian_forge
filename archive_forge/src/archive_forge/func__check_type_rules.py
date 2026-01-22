from contextlib import contextmanager
from typing import Dict, List
def _check_type_rules(self, node):
    for rule in self._rule_type_instances.get(node.type, []):
        rule.feed_node(node)