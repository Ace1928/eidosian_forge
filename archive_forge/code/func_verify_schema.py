import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def verify_schema(self, from_node_labels: List[str], relation_types: List[str], to_node_labels: List[str]) -> bool:
    """
        Args:
            from_node_labels: labels of the from node
            relation_type: type of the relation
            to_node_labels: labels of the to node
        """
    valid_schemas = self.schemas
    if from_node_labels != []:
        from_node_labels = [label.strip('`') for label in from_node_labels]
        valid_schemas = [schema for schema in valid_schemas if schema[0] in from_node_labels]
    if to_node_labels != []:
        to_node_labels = [label.strip('`') for label in to_node_labels]
        valid_schemas = [schema for schema in valid_schemas if schema[2] in to_node_labels]
    if relation_types != []:
        relation_types = [type.strip('`') for type in relation_types]
        valid_schemas = [schema for schema in valid_schemas if schema[1] in relation_types]
    return valid_schemas != []