import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
class LinkCallback:
    source_model = None
    target_model = None
    source_handles = []
    target_handles = []
    on_source_events = []
    on_source_changes = []
    on_target_events = []
    on_target_changes = []
    source_code = None
    target_code = None

    def __init__(self, root_model, link, source_plot, target_plot=None):
        self.root_model = root_model
        self.link = link
        self.source_plot = source_plot
        self.target_plot = target_plot
        self.validate()
        references = {k: v for k, v in link.param.values().items() if k not in ('source', 'target', 'name')}
        for sh in self.source_handles + [self.source_model]:
            key = f'source_{sh}'
            references[key] = source_plot.handles[sh]
        for p, value in link.param.values().items():
            if p in ('name', 'source', 'target'):
                continue
            references[p] = value
        if target_plot is not None:
            for sh in self.target_handles + [self.target_model]:
                key = f'target_{sh}'
                references[key] = target_plot.handles[sh]
        if self.source_model in source_plot.handles:
            src_model = source_plot.handles[self.source_model]
            src_cb = CustomJS(args=references, code=self.source_code)
            for ch in self.on_source_changes:
                src_model.js_on_change(ch, src_cb)
            for ev in self.on_source_events:
                src_model.js_on_event(ev, src_cb)
            self.src_cb = src_cb
        else:
            self.src_cb = None
        if target_plot is not None and self.target_model in target_plot.handles and self.target_code:
            tgt_model = target_plot.handles[self.target_model]
            tgt_cb = CustomJS(args=references, code=self.target_code)
            for ch in self.on_target_changes:
                tgt_model.js_on_change(ch, tgt_cb)
            for ev in self.on_target_events:
                tgt_model.js_on_event(ev, tgt_cb)
            self.tgt_cb = tgt_cb
        else:
            self.tgt_cb = None

    @classmethod
    def find_links(cls, root_plot):
        """
        Traverses the supplied plot and searches for any Links on
        the plotted objects.
        """
        plot_fn = lambda x: isinstance(x, (GenericElementPlot, GenericOverlayPlot))
        plots = root_plot.traverse(lambda x: x, [plot_fn])
        potentials = [cls.find_link(plot) for plot in plots]
        source_links = [p for p in potentials if p is not None]
        found = []
        for plot, links in source_links:
            for link in links:
                if not link._requires_target:
                    found.append((link, plot, None))
                    continue
                potentials = [cls.find_link(p, link) for p in plots]
                tgt_links = [p for p in potentials if p is not None]
                if tgt_links:
                    found.append((link, plot, tgt_links[0][0]))
        return found

    @classmethod
    def find_link(cls, plot, link=None):
        """
        Searches a GenericElementPlot for a Link.
        """
        registry = Link.registry.items()
        for source in plot.link_sources:
            if link is None:
                links = [l for src, links in registry for l in links if src is source or (src._plot_id is not None and src._plot_id == source._plot_id)]
                if links:
                    return (plot, links)
            elif link.target is source or (link.target is not None and link.target._plot_id is not None and (link.target._plot_id == source._plot_id)):
                return (plot, [link])

    def validate(self):
        """
        Should be subclassed to check if the source and target plots
        are compatible to perform the linking.
        """