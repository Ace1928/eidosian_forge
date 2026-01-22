import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
def _get_triples(self, e_labels: List[str]) -> List[str]:
    triple_query = '\n        MATCH (a)-[e:`{e_label}`]->(b)\n        WITH a,e,b LIMIT 3000\n        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to\n        LIMIT 10\n        '
    triple_template = '(:`{a}`)-[:`{e}`]->(:`{b}`)'
    triple_schema = []
    for label in e_labels:
        q = triple_query.format(e_label=label)
        data = self.query(q)
        for d in data:
            triple = triple_template.format(a=d['from'][0], e=d['edge'], b=d['to'][0])
            triple_schema.append(triple)
    return triple_schema