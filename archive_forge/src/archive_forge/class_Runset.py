import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@dataclass(config=dataclass_config)
class Runset(Base):
    entity: str = ''
    project: str = ''
    name: str = 'Run set'
    query: str = ''
    filters: Optional[str] = ''
    groupby: LList[str] = Field(default_factory=list)
    order: LList[OrderBy] = Field(default_factory=lambda: [OrderBy('CreatedTimestamp', ascending=False)])
    custom_run_colors: Dict[Union[str, Tuple[MetricType, ...]], str] = Field(default_factory=dict)
    _id: str = Field(default_factory=internal._generate_name, init=False, repr=False)

    def to_model(self):
        project = None
        if self.entity or self.project:
            project = internal.Project(entity_name=self.entity, name=self.project)
        obj = internal.Runset(project=project, name=self.name, filters=expr_parsing.expr_to_filters(self.filters), grouping=[internal.Key(name=expr_parsing.to_backend_name(g)) for g in self.groupby], sort=internal.Sort(keys=[o.to_model() for o in self.order]))
        obj.id = self._id
        return obj

    @classmethod
    def from_model(cls, model: internal.Runset):
        entity = ''
        project = ''
        p = model.project
        if p is not None:
            if p.entity_name:
                entity = p.entity_name
            if p.name:
                project = p.name
        obj = cls(entity=entity, project=project, name=model.name, filters=expr_parsing.filters_to_expr(model.filters), groupby=[expr_parsing.to_frontend_name(k.name) for k in model.grouping], order=[OrderBy.from_model(s) for s in model.sort.keys])
        obj._id = model.id
        return obj