from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
def _merge_es_range_queries(self, conditions: List[Dict]) -> List[Dict[str, Dict]]:
    """
        Merges Elasticsearch range queries that perform on the same metadata field.
        """
    range_conditions = [cond['range'] for cond in filter(lambda condition: 'range' in condition, conditions)]
    if range_conditions:
        conditions = [condition for condition in conditions if 'range' not in condition]
        range_conditions_dict = nested_defaultdict()
        for condition in range_conditions:
            field_name = list(condition.keys())[0]
            operation = list(condition[field_name].keys())[0]
            comparison_value = condition[field_name][operation]
            range_conditions_dict[field_name][operation] = comparison_value
        conditions.extend(({'range': {field_name: comparison_operations}} for field_name, comparison_operations in range_conditions_dict.items()))
    return conditions