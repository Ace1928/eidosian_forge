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
class ScalarChart(Panel):
    title: Optional[str] = None
    metric: MetricType = ''
    groupby_aggfunc: Optional[GroupAgg] = None
    groupby_rangefunc: Optional[GroupArea] = None
    custom_expressions: Optional[LList[str]] = None
    legend_template: Optional[str] = None
    font_size: Optional[FontSize] = None

    def to_model(self):
        obj = internal.ScalarChart(config=internal.ScalarChartConfig(chart_title=self.title, metrics=[_metric_to_backend(self.metric)], group_agg=self.groupby_aggfunc, group_area=self.groupby_rangefunc, expressions=self.custom_expressions, legend_template=self.legend_template, font_size=self.font_size), layout=self.layout.to_model(), id=self.id)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.ScatterPlot):
        obj = cls(title=model.config.chart_title, metric=_metric_to_frontend(model.config.metrics[0]), groupby_aggfunc=model.config.group_agg, groupby_rangefunc=model.config.group_area, custom_expressions=model.config.expressions, legend_template=model.config.legend_template, font_size=model.config.font_size, layout=Layout.from_model(model.layout), id=model.id)
        obj._ref = model.ref
        return obj