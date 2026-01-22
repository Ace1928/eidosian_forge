from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        return all((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'must': conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(and_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {'operator': 'And', 'operands': conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {'$and': conditions}

    def invert(self) -> 'OrOperation':
        return OrOperation([condition.invert() for condition in self.conditions])