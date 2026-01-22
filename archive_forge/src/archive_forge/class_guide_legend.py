from __future__ import annotations
import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import remove_missing
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from .guide import GuideElements, guide
@dataclass
class guide_legend(guide):
    """
    Legend guide
    """
    nrow: Optional[int] = None
    'Number of rows of legends.'
    ncol: Optional[int] = None
    'Number of columns of legends.'
    byrow: bool = False
    'Whether to fill the legend row-wise or column-wise.'
    override_aes: dict[str, Any] = field(default_factory=dict)
    'Aesthetic parameters of legend key.'
    available_aes: set[str] = field(init=False, default_factory=lambda: {'any'})
    'Aesthetics for which this guide can be used'
    _layer_parameters: list[LayerParameters] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._elements_cls = GuideElementsLegend
        self.elements: GuideElementsLegend

    def train(self, scale, aesthetic=None):
        """
        Create the key for the guide

        The key is a dataframe with two columns:

        - scale name : values
        - label : labels for each value

        scale name is one of the aesthetics: `x`, `y`, `color`,
        `fill`, `size`, `shape`, `alpha`, `stroke`.

        Returns this guide if training is successful and None
        if it fails
        """
        if aesthetic is None:
            aesthetic = scale.aesthetics[0]
        breaks = scale.get_bounded_breaks()
        if not breaks:
            return None
        key = pd.DataFrame({aesthetic: scale.map(breaks), 'label': scale.get_labels(breaks)})
        if len(key) == 0:
            return None
        self.key = key
        labels = ' '.join((str(x) for x in self.key['label']))
        info = '\n'.join([str(self.title), labels, str(self.direction), self.__class__.__name__])
        self.hash = hashlib.sha256(info.encode('utf-8')).hexdigest()
        return self

    def merge(self, other):
        """
        Merge overlapped guides

        For example:

        ```python
        from ggplot import *
        p = (
            ggplot(aes(x="cut", fill="cut", color="cut"), data=diamonds)
            + stat_bin()
        )
        ```

        Would create similar guides for fill and color where only
        a single guide would do
        """
        self.key = self.key.merge(other.key)
        duplicated = set(self.override_aes) & set(other.override_aes)
        if duplicated:
            msg = f'Duplicated override_aes, `{duplicated}`, are  ignored.'
            warn(msg, PlotnineWarning)
        self.override_aes.update(other.override_aes)
        for ae in duplicated:
            del self.override_aes[ae]
        return self

    def create_geoms(self):
        """
        Make information needed to draw a legend for each of the layers.

        For each layer, that information is a dictionary with the geom
        to draw the guide together with the data and the parameters that
        will be used in the call to geom.
        """
        for l in self.plot_layers:
            exclude = set()
            if isinstance(l.show_legend, dict):
                l.show_legend = rename_aesthetics(l.show_legend)
                exclude = {ae for ae, val in l.show_legend.items() if not val}
            elif l.show_legend not in (None, True):
                continue
            matched = self.legend_aesthetics(l)
            if not set(matched) - exclude:
                continue
            data = self.key[matched].copy()
            try:
                data = l.use_defaults(data)
            except PlotnineError:
                warn('Failed to apply `after_scale` modifications to the legend.', PlotnineWarning)
                data = l.use_defaults(data, aes_modifiers={})
            for ae in set(self.override_aes) & set(data.columns):
                data[ae] = self.override_aes[ae]
            data = remove_missing(data, l.geom.params['na_rm'], list(l.geom.REQUIRED_AES | l.geom.NON_MISSING_AES), f'{l.geom.__class__.__name__} legend')
            self._layer_parameters.append(LayerParameters(l.geom, data, l))
        if not self._layer_parameters:
            return None
        return self

    def _calculate_rows_and_cols(self, elements: GuideElementsLegend) -> TupleInt2:
        nrow, ncol = (self.nrow, self.ncol)
        nbreak = len(self.key)
        if nrow and ncol:
            if nrow * ncol < nbreak:
                raise PlotnineError('nrow x ncol needs to be larger than the number of breaks')
            return (nrow, ncol)
        if (nrow, ncol) == (None, None):
            if elements.is_horizontal:
                nrow = int(np.ceil(nbreak / 5))
            else:
                ncol = int(np.ceil(nbreak / 20))
        if nrow is None:
            ncol = cast(int, ncol)
            nrow = int(np.ceil(nbreak / ncol))
        elif ncol is None:
            nrow = cast(int, nrow)
            ncol = int(np.ceil(nbreak / nrow))
        return (nrow, ncol)

    def draw(self):
        """
        Draw guide

        Returns
        -------
        out : matplotlib.offsetbox.Offsetbox
            A drawing of this legend
        """
        from matplotlib.offsetbox import HPacker, TextArea, VPacker
        from .._mpl.offsetbox import ColoredDrawingArea
        obverse = slice(0, None)
        reverse = slice(None, None, -1)
        nbreak = len(self.key)
        targets = self.theme.targets
        keys_order = reverse if self.reverse else obverse
        elements = self.elements
        title = cast(str, self.title)
        title_box = TextArea(title)
        targets.legend_title = title_box._text
        props = {'ha': elements.text.ha, 'va': elements.text.va}
        labels = [TextArea(s, textprops=props) for s in self.key['label']]
        _texts = [l._text for l in labels]
        targets.legend_text_legend = _texts
        drawings: list[ColoredDrawingArea] = []
        for i in range(nbreak):
            da = ColoredDrawingArea(elements.key_widths[i], elements.key_heights[i], 0, 0)
            for params in self._layer_parameters:
                with suppress(IndexError):
                    key_data = params.data.iloc[i]
                    params.geom.draw_legend(key_data, da, params.layer)
            drawings.append(da)
        targets.legend_key = drawings
        lookup: dict[SidePosition, tuple[type[PackerBase], slice]] = {'right': (HPacker, reverse), 'left': (HPacker, obverse), 'bottom': (VPacker, reverse), 'top': (VPacker, obverse)}
        packer, slc = lookup[elements.text_position]
        if self.elements.text.is_blank:
            key_boxes = [d for d in drawings][keys_order]
        else:
            key_boxes = [packer(children=[l, d][slc], sep=elements.text.margin, align=elements.text.align, pad=0) for d, l in zip(drawings, labels)][keys_order]
        nrow, ncol = self._calculate_rows_and_cols(elements)
        if self.byrow:
            chunk_size = ncol
            packer_dim1, packer_dim2 = (HPacker, VPacker)
            sep1 = elements.key_spacing_x
            sep2 = elements.key_spacing_y
        else:
            chunk_size = nrow
            packer_dim1, packer_dim2 = (VPacker, HPacker)
            sep1 = elements.key_spacing_y
            sep2 = elements.key_spacing_x
        chunks = []
        for i in range(len(key_boxes)):
            start = i * chunk_size
            stop = start + chunk_size
            s = islice(key_boxes, start, stop)
            chunks.append(list(s))
            if stop >= len(key_boxes):
                break
        chunk_boxes: list[Artist] = [packer_dim1(children=chunk, align='left', sep=sep1, pad=0) for chunk in chunks]
        entries_box = packer_dim2(children=chunk_boxes, align='baseline', sep=sep2, pad=0)
        packer, slc = lookup[elements.title_position]
        if elements.title.is_blank:
            children: list[Artist] = [entries_box]
        else:
            children = [title_box, entries_box][slc]
        box = packer(children=children, sep=elements.title.margin, align=elements.title.align, pad=elements.margin)
        return box