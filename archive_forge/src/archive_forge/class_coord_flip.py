from __future__ import annotations
import typing
import pandas as pd
from ..iapi import labels_view, panel_ranges, panel_view
from .coord_cartesian import coord_cartesian
class coord_flip(coord_cartesian):
    """
    Flipped cartesian coordinates

    The horizontal becomes vertical, and vertical becomes horizontal.
    This is primarily useful for converting geoms and statistics which
    display y conditional on x, to x conditional on y.

    Parameters
    ----------
    xlim : tuple[float, float], default=None
        Limits for x axis. If None, then they are automatically computed.
    ylim : tuple[float, float], default=None
        Limits for y axis. If None, then they are automatically computed.
    expand : bool, default=True
        If `True`, expand the coordinate axes by some factor. If `False`,
        use the limits from the data.
    """

    def labels(self, cur_labels: labels_view) -> labels_view:
        return flip_labels(super().labels(cur_labels))

    def transform(self, data: pd.DataFrame, panel_params: panel_view, munch: bool=False) -> pd.DataFrame:
        data = flip_labels(data)
        return super().transform(data, panel_params, munch=munch)

    def setup_panel_params(self, scale_x: scale, scale_y: scale) -> panel_view:
        panel_params = super().setup_panel_params(scale_x, scale_y)
        return flip_labels(panel_params)

    def setup_layout(self, layout: pd.DataFrame) -> pd.DataFrame:
        x, y = ('SCALE_X', 'SCALE_Y')
        layout[x], layout[y] = (layout[y].copy(), layout[x].copy())
        return layout

    def range(self, panel_params: panel_view) -> panel_ranges:
        """
        Return the range along the dimensions of the coordinate system
        """
        return panel_ranges(x=panel_params.y.range, y=panel_params.x.range)