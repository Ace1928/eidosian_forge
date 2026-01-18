from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
@staticmethod
def user_query(table_key: str) -> Dict[str, Any]:
    return {'queryFields': [{'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}], 'fields': [{'name': 'id', 'fields': []}, {'name': 'name', 'fields': []}, {'name': '_defaultColorIndex', 'fields': []}, {'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': table_key}], 'fields': []}]}]}