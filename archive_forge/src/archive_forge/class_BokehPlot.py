from collections import defaultdict
from itertools import groupby
import numpy as np
import param
from bokeh.layouts import gridplot
from bokeh.models import (
from bokeh.models.layouts import TabPanel, Tabs
from ...core import (
from ...core.options import SkipRendering
from ...core.util import (
from ...selection import NoOpSelectionDisplay
from ..links import Link
from ..plot import (
from ..util import attach_streams, collate, displayable
from .links import LinkCallback
from .util import (
class BokehPlot(DimensionedPlot, CallbackPlot):
    """
    Plotting baseclass for the Bokeh backends, implementing the basic
    plotting interface for Bokeh based plots.
    """
    shared_datasource = param.Boolean(default=True, doc='\n        Whether Elements drawing the data from the same object should\n        share their Bokeh data source allowing for linked brushing\n        and other linked behaviors.')
    title = param.String(default='{label} {group} {dimensions}', doc='\n        The formatting string for the title of this plot, allows defining\n        a label group separator and dimension labels.')
    title_format = param.String(default=None, doc='Alias for title.')
    toolbar = param.ObjectSelector(default='above', objects=['above', 'below', 'left', 'right', None], doc="\n        The toolbar location, must be one of 'above', 'below',\n        'left', 'right', None.")
    width = param.Integer(default=None, bounds=(0, None), doc='\n        The width of the component (in pixels). This can be either\n        fixed or preferred width, depending on width sizing policy.')
    height = param.Integer(default=None, bounds=(0, None), doc='\n        The height of the component (in pixels).  This can be either\n        fixed or preferred height, depending on height sizing policy.')
    _merged_tools = ['pan', 'box_zoom', 'box_select', 'lasso_select', 'poly_select', 'ypan', 'xpan']
    _title_template = '<span style="color:{color};font-family:{font};font-style:{fontstyle};font-weight:{fontstyle};font-size:{fontsize}">{title}</span>'
    backend = 'bokeh'
    selection_display = NoOpSelectionDisplay()

    @property
    def id(self):
        return self.root.ref['id'] if self.root else None

    def get_data(self, element, ranges, style):
        """
        Returns the data from an element in the appropriate format for
        initializing or updating a ColumnDataSource and a dictionary
        which maps the expected keywords arguments of a glyph to
        the column in the datasource.
        """
        raise NotImplementedError

    def _update_selected(self, cds):
        from .callbacks import Selection1DCallback
        cds.selected.indices = self.selected
        for cb in self.callbacks:
            if isinstance(cb, Selection1DCallback):
                for s in cb.streams:
                    s.update(index=self.selected)

    def _init_datasource(self, data):
        """
        Initializes a data source to be passed into the bokeh glyph.
        """
        data = self._postprocess_data(data)
        cds = ColumnDataSource(data=data)
        if hasattr(self, 'selected') and self.selected is not None:
            self._update_selected(cds)
        return cds

    def _postprocess_data(self, data):
        """
        Applies necessary type transformation to the data before
        it is set on a ColumnDataSource.
        """
        new_data = {}
        for k, values in data.items():
            values = decode_bytes(values)
            if len(values) and isinstance(values[0], cftime_types):
                if any((v.calendar not in _STANDARD_CALENDARS for v in values)):
                    self.param.warning('Converting cftime.datetime from a non-standard calendar (%s) to a standard calendar for plotting. This may lead to subtle errors in formatting dates, for accurate tick formatting switch to the matplotlib backend.' % values[0].calendar)
                values = cftime_to_timestamp(values, 'ms')
            new_data[k] = values
        return new_data

    def _update_datasource(self, source, data):
        """
        Update datasource with data for a new frame.
        """
        if not self.document:
            return
        data = self._postprocess_data(data)
        empty = all((len(v) == 0 for v in data.values()))
        if self.streaming and self.streaming[0].data is self.current_frame.data and self._stream_data and (not empty):
            stream = self.streaming[0]
            if stream._triggering:
                data = {k: v[-stream._chunk_length:] for k, v in data.items()}
                source.stream(data, stream.length)
            return
        if cds_column_replace(source, data):
            source.data = data
        else:
            source.data.update(data)
        if hasattr(self, 'selected') and self.selected is not None:
            self._update_selected(source)

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        return self.handles['plot']

    @property
    def current_handles(self):
        """
        Should return a list of plot objects that have changed and
        should be updated.
        """
        return []

    def _get_fontsize_defaults(self):
        theme = self.renderer.theme
        defaults = {'title': get_default(Title, 'text_font_size', theme), 'legend_title': get_default(Legend, 'title_text_font_size', theme), 'legend': get_default(Legend, 'label_text_font_size', theme), 'label': get_default(Axis, 'axis_label_text_font_size', theme), 'ticks': get_default(Axis, 'major_label_text_font_size', theme), 'cticks': get_default(ColorBar, 'major_label_text_font_size', theme), 'clabel': get_default(ColorBar, 'title_text_font_size', theme)}
        processed = dict(defaults)
        for k, v in defaults.items():
            if isinstance(v, dict) and 'value' in v:
                processed[k] = v['value']
        return processed

    def cleanup(self):
        """
        Cleans up references to the plot after the plot has been
        deleted. Traverses through all plots cleaning up Callbacks and
        Stream subscribers.
        """
        plots = self.traverse(lambda x: x, [BokehPlot])
        for plot in plots:
            if not isinstance(plot, (GenericCompositePlot, GenericElementPlot, GenericOverlayPlot)):
                continue
            streams = list(plot.streams)
            plot.streams = []
            plot._document = None
            if plot.subplots:
                plot.subplots.clear()
            if isinstance(plot, GenericElementPlot):
                for callback in plot.callbacks:
                    streams += callback.streams
                    callback.cleanup()
            for stream in set(streams):
                stream._subscribers = [(p, subscriber) for p, subscriber in stream._subscribers if not is_param_method(subscriber) or get_method_owner(subscriber) not in plots]

    def _fontsize(self, key, label='fontsize', common=True):
        """
        Converts integer fontsizes to a string specifying
        fontsize in pt.
        """
        size = super()._fontsize(key, label, common)
        return {k: v if isinstance(v, str) else f'{v}pt' for k, v in size.items()}

    def _get_title_div(self, key, default_fontsize='15pt', width=450):
        title_div = None
        title = self._format_title(key) if self.show_title else ''
        if not title:
            return title_div
        title_json = theme_attr_json(self.renderer.theme, 'Title')
        color = title_json.get('text_color', None)
        font = title_json.get('text_font', 'Arial')
        fontstyle = title_json.get('text_font_style', 'bold')
        fontsize = self._fontsize('title').get('fontsize', default_fontsize)
        if fontsize == default_fontsize:
            fontsize = title_json.get('text_font_size', default_fontsize)
            if 'em' in fontsize:
                fontsize = str(float(fontsize[:-2]) + 0.25) + 'em'
        title_tags = self._title_template.format(color=color, font=font, fontstyle=fontstyle, fontsize=fontsize, title=title)
        if 'title' in self.handles:
            title_div = self.handles['title']
        else:
            title_div = Div(width=width, styles={'white-space': 'nowrap'})
        title_div.text = title_tags
        return title_div

    def sync_sources(self):
        """
        Syncs data sources between Elements, which draw data
        from the same object.
        """
        get_sources = lambda x: (id(x.current_frame.data), x)
        filter_fn = lambda x: x.shared_datasource and x.current_frame is not None and (not isinstance(x.current_frame.data, np.ndarray)) and ('source' in x.handles)
        data_sources = self.traverse(get_sources, [filter_fn])
        grouped_sources = groupby(sorted(data_sources, key=lambda x: x[0]), lambda x: x[0])
        shared_sources = []
        source_cols = {}
        plots = []
        for _, group in grouped_sources:
            group = list(group)
            if len(group) > 1:
                source_data = {}
                for _, plot in group:
                    source_data.update(plot.handles['source'].data)
                new_source = ColumnDataSource(source_data)
                for _, plot in group:
                    renderer = plot.handles.get('glyph_renderer')
                    for callback in plot.callbacks:
                        callback.reset()
                    if renderer is None:
                        continue
                    elif 'data_source' in renderer.properties():
                        renderer.update(data_source=new_source)
                    else:
                        renderer.update(source=new_source)
                    plot.handles['source'] = plot.handles['cds'] = new_source
                    plots.append(plot)
                shared_sources.append(new_source)
                source_cols[id(new_source)] = [c for c in new_source.data]
        for plot in plots:
            for hook in plot.hooks:
                hook(plot, plot.current_frame)
            for callback in plot.callbacks:
                callback.initialize(plot_id=self.id)
        self.handles['shared_sources'] = shared_sources
        self.handles['source_cols'] = source_cols

    def init_links(self):
        links = LinkCallback.find_links(self)
        callbacks = []
        for link, src_plot, tgt_plot in links:
            cb = Link._callbacks['bokeh'][type(link)]
            if src_plot is None or (link._requires_target and tgt_plot is None):
                continue
            callbacks.append(cb(self.root, link, src_plot, tgt_plot))
        return callbacks