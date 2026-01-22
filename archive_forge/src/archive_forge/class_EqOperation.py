from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Union[str, int, float, bool]]]:
        assert not isinstance(self.comparison_value, list), "Use '$in' operation for lists as comparison values."
        return {'term': {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value == self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        return {'path': [self.field_name], 'operator': 'Equal', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {'$eq': self.comparison_value}}

    def invert(self) -> 'NeOperation':
        return NeOperation(self.field_name, self.comparison_value)