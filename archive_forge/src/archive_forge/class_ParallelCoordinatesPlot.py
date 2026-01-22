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
class ParallelCoordinatesPlot(Panel):
    columns: LList[ParallelCoordinatesPlotColumn] = Field(default_factory=list)
    title: Optional[str] = None
    gradient: Optional[LList[GradientPoint]] = None
    font_size: Optional[FontSize] = None

    def to_model(self):
        gradient = self.gradient
        if gradient is not None:
            gradient = [x.to_model() for x in self.gradient]
        obj = internal.ParallelCoordinatesPlot(config=internal.ParallelCoordinatesPlotConfig(chart_title=self.title, columns=[c.to_model() for c in self.columns], custom_gradient=gradient, font_size=self.font_size), layout=self.layout.to_model(), id=self.id)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.ScatterPlot):
        gradient = model.config.custom_gradient
        if gradient is not None:
            gradient = [GradientPoint.from_model(x) for x in gradient]
        obj = cls(columns=[ParallelCoordinatesPlotColumn.from_model(c) for c in model.config.columns], title=model.config.chart_title, gradient=gradient, font_size=model.config.font_size, layout=Layout.from_model(model.layout), id=model.id)
        obj._ref = model.ref
        return obj