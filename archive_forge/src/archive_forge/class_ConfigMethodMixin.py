import sys
from . import core
from altair.utils import use_signature
from altair.utils.schemapi import Undefined, UndefinedType
from typing import Any, Sequence, List, Literal, Union
class ConfigMethodMixin:
    """A mixin class that defines config methods"""

    @use_signature(core.Config)
    def configure(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=False)
        copy.config = core.Config(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_arc(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['arc'] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.AreaConfig)
    def configure_area(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['area'] = core.AreaConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axis(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axis'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisBand(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisBand'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisBottom(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisBottom'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisDiscrete(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisDiscrete'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisLeft(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisLeft'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisPoint(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisPoint'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisQuantitative(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisQuantitative'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisRight(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisRight'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisTemporal(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisTemporal'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisTop(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisTop'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisX(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisX'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXBand(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisXBand'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXDiscrete(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisXDiscrete'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXPoint(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisXPoint'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXQuantitative(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisXQuantitative'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXTemporal(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisXTemporal'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisY(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisY'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYBand(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisYBand'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYDiscrete(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisYDiscrete'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYPoint(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisYPoint'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYQuantitative(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisYQuantitative'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYTemporal(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['axisYTemporal'] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.BarConfig)
    def configure_bar(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['bar'] = core.BarConfig(*args, **kwargs)
        return copy

    @use_signature(core.BoxPlotConfig)
    def configure_boxplot(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['boxplot'] = core.BoxPlotConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_circle(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['circle'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.CompositionConfig)
    def configure_concat(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['concat'] = core.CompositionConfig(*args, **kwargs)
        return copy

    @use_signature(core.ErrorBandConfig)
    def configure_errorband(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['errorband'] = core.ErrorBandConfig(*args, **kwargs)
        return copy

    @use_signature(core.ErrorBarConfig)
    def configure_errorbar(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['errorbar'] = core.ErrorBarConfig(*args, **kwargs)
        return copy

    @use_signature(core.CompositionConfig)
    def configure_facet(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['facet'] = core.CompositionConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_geoshape(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['geoshape'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_header(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['header'] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerColumn(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['headerColumn'] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerFacet(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['headerFacet'] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerRow(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['headerRow'] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_image(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['image'] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.LegendConfig)
    def configure_legend(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['legend'] = core.LegendConfig(*args, **kwargs)
        return copy

    @use_signature(core.LineConfig)
    def configure_line(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['line'] = core.LineConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_mark(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['mark'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_point(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['point'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.ProjectionConfig)
    def configure_projection(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['projection'] = core.ProjectionConfig(*args, **kwargs)
        return copy

    @use_signature(core.RangeConfig)
    def configure_range(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['range'] = core.RangeConfig(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_rect(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['rect'] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_rule(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['rule'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.ScaleConfig)
    def configure_scale(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['scale'] = core.ScaleConfig(*args, **kwargs)
        return copy

    @use_signature(core.SelectionConfig)
    def configure_selection(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['selection'] = core.SelectionConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_square(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['square'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_text(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['text'] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.TickConfig)
    def configure_tick(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['tick'] = core.TickConfig(*args, **kwargs)
        return copy

    @use_signature(core.TitleConfig)
    def configure_title(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['title'] = core.TitleConfig(*args, **kwargs)
        return copy

    @use_signature(core.FormatConfig)
    def configure_tooltipFormat(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['tooltipFormat'] = core.FormatConfig(*args, **kwargs)
        return copy

    @use_signature(core.LineConfig)
    def configure_trail(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['trail'] = core.LineConfig(*args, **kwargs)
        return copy

    @use_signature(core.ViewConfig)
    def configure_view(self, *args, **kwargs) -> Self:
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config['view'] = core.ViewConfig(*args, **kwargs)
        return copy