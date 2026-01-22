import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
def _get_node_properties(self, n_labels: List[str], types: Dict) -> List:
    node_properties_query = '\n        MATCH (a:`{n_label}`)\n        RETURN properties(a) AS props\n        LIMIT 100\n        '
    node_properties = []
    for label in n_labels:
        q = node_properties_query.format(n_label=label)
        data = {'label': label, 'properties': self.query(q)}
        s = set({})
        for p in data['properties']:
            for k, v in p['props'].items():
                s.add((k, types[type(v).__name__]))
        np = {'properties': [{'property': k, 'type': v} for k, v in s], 'labels': label}
        node_properties.append(np)
    return node_properties