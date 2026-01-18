import json
import logging
import sqlite3
from typing import List, Any, Dict, Tuple
def transform_and_fill_data(input_data: Dict, template: Dict) -> Tuple[Dict, List[str]]:
    transformed_data = {}
    related_topics = []
    for key in template.keys():
        if key in input_data:
            transformed_data[key] = input_data[key]
        else:
            search_results = mock_web_search(key)
            related_topics.extend(search_results)
            transformed_data[key] = search_results[0] if search_results else None
    return (transformed_data, related_topics)