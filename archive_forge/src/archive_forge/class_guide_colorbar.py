from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
@dataclass
class guide_colorbar(guide):
    """
    Guide colorbar

    Notes
    -----
    To correctly place a rasterized colorbar when saving the plot as an `svg`
    or `pdf`, you should set the `dpi` to 72 i.e. `theme(dpi=72)`{.py}.
    """
    nbin: Optional[int] = None
    '\n    Number of bins for drawing a colorbar. A larger value yields\n    a smoother colorbar\n    '
    display: Literal['gradient', 'rectangles', 'raster'] = 'gradient'
    'How to render the colorbar.'
    alpha: Optional[float] = None
    '\n    Opacity (in the range `[0, 1]`) of the colorbar. The defualt\n    `None`, is to use the opacity of the plot.\n    '
    draw_ulim: bool = True
    'Whether to show the upper limit tick marks.'
    draw_llim: bool = True
    'Whether to show the lower limit tick marks. '
    available_aes: set[str] = field(init=False, default_factory=lambda: {'colour', 'color', 'fill'})

    def __post_init__(self):
        self._elements_cls = GuideElementsColorbar
        self.elements: GuideElementsColorbar
        if self.nbin is None:
            self.nbin = 300

    def train(self, scale: scale, aesthetic=None):
        self.nbin = cast(int, self.nbin)
        self.title = cast(str, self.title)
        if not isinstance(scale, scale_continuous):
            warn('colorbar guide needs continuous scales', PlotnineWarning)
            return None
        if aesthetic is None:
            aesthetic = scale.aesthetics[0]
        if set(scale.aesthetics) & self.available_aes == 0:
            warn('colorbar guide needs appropriate scales.', PlotnineWarning)
            return None
        limits = scale.limits
        breaks = scale.get_bounded_breaks()
        if not len(breaks):
            return None
        self.key = pd.DataFrame({aesthetic: scale.map(breaks), 'label': scale.get_labels(breaks), 'value': breaks})
        bar = np.linspace(limits[0], limits[1], self.nbin)
        self.bar = pd.DataFrame({'color': scale.map(bar), 'value': bar})
        labels = ' '.join((str(x) for x in self.key['label']))
        info = '\n'.join([self.title, labels, ' '.join(self.bar['color'].tolist()), self.__class__.__name__])
        self.hash = hashlib.sha256(info.encode('utf-8')).hexdigest()
        return self

    def merge(self, other):
        """
        Simply discards the other guide
        """
        return self

    def create_geoms(self):
        """
        Return self if colorbar will be drawn and None if not

        This guide is not geom based
        """
        for l in self.plot_layers:
            exclude = set()
            if isinstance(l.show_legend, dict):
                l.show_legend = rename_aesthetics(l.show_legend)
                exclude = {ae for ae, val in l.show_legend.items() if not val}
            elif l.show_legend not in (None, True):
                continue
            matched = self.legend_aesthetics(l)
            if set(matched) - exclude:
                break
        else:
            return None
        return self

    def draw(self):
        """
        Draw guide

        Returns
        -------
        out : matplotlib.offsetbox.Offsetbox
            A drawing of this legend
        """
        from matplotlib.offsetbox import HPacker, TextArea, VPacker
        from matplotlib.transforms import IdentityTransform
        from .._mpl.offsetbox import DPICorAuxTransformBox
        self.theme = cast('theme', self.theme)
        obverse = slice(0, None)
        reverse = slice(None, None, -1)
        nbars = len(self.bar)
        elements = self.elements
        raster = self.display == 'raster'
        colors = self.bar['color'].tolist()
        labels = self.key['label'].tolist()
        targets = self.theme.targets
        _from = (self.bar['value'].min(), self.bar['value'].max())
        tick_locations = rescale(self.key['value'], (0.5, nbars - 0.5), _from) * elements.key_height / nbars
        if nbars >= 150 and len(tick_locations) >= 2:
            tick_locations = [np.floor(tick_locations[0]), *np.round(tick_locations[1:-1]), np.ceil(tick_locations[-1])]
        if self.reverse:
            colors = colors[::-1]
            labels = labels[::-1]
            tick_locations = elements.key_height - tick_locations[::-1]
        auxbox = DPICorAuxTransformBox(IdentityTransform())
        title = cast(str, self.title)
        props = {'ha': elements.title.ha, 'va': elements.title.va}
        title_box = TextArea(title, textprops=props)
        targets.legend_title = title_box._text
        if not self.elements.text.is_blank:
            texts = add_labels(auxbox, labels, tick_locations, elements)
            targets.legend_text_colorbar = texts
        if self.display == 'rectangles':
            add_segmented_colorbar(auxbox, colors, elements)
        else:
            add_gradient_colorbar(auxbox, colors, elements, raster)
        visible = slice(None if self.draw_llim else 1, None if self.draw_ulim else -1)
        coll = add_ticks(auxbox, tick_locations[visible], elements)
        targets.legend_ticks = coll
        frame = add_frame(auxbox, elements)
        targets.legend_frame = frame
        lookup: dict[SidePosition, tuple[type[PackerBase], slice]] = {'right': (HPacker, reverse), 'left': (HPacker, obverse), 'bottom': (VPacker, reverse), 'top': (VPacker, obverse)}
        packer, slc = lookup[elements.title_position]
        if elements.title.is_blank:
            children: list[Artist] = [auxbox]
        else:
            children = [title_box, auxbox][slc]
        box = packer(children=children, sep=elements.title.margin, align=elements.title.align, pad=0)
        return box