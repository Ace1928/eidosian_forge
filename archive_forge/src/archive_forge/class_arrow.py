from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
class arrow:
    """
    Define arrow (actually an arrowhead)

    This is used to define arrow heads for
    [](`~plotnine.geoms.geom_path`).

    Parameters
    ----------
    angle :
        angle in degrees between the tail a
        single edge.
    length :
        of the edge in "inches"
    ends :
        At which end of the line to draw the
        arrowhead
    type :
        When it is closed, it is also filled
    """

    def __init__(self, angle: float=30, length: float=0.2, ends: Literal['first', 'last', 'both']='last', type: Literal['open', 'closed']='open'):
        self.angle = angle
        self.length = length
        self.ends = ends
        self.type = type

    def draw(self, data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, constant: bool=True, **params: Any):
        """
        Draw arrows at the end(s) of the lines

        Parameters
        ----------
        data : dataframe
            Data to be plotted by this geom. This is the
            dataframe created in the plot_build pipeline.
        panel_params : panel_view
            The scale information as may be required by the
            axes. At this point, that information is about
            ranges, ticks and labels. Attributes are of interest
            to the geom are:

            ```python
            "panel_params.x.range"  # tuple
            "panel_params.y.range"  # tuple
            ```
        coord : coord
            Coordinate (e.g. coord_cartesian) system of the
            geom.
        ax : axes
            Axes on which to plot.
        constant: bool
            If the path attributes vary along the way. If false,
            the arrows are per segment of the path
        params : dict
            Combined parameters for the geom and stat. Also
            includes the `zorder`.
        """
        first = self.ends in ('first', 'both')
        last = self.ends in ('last', 'both')
        data = data.sort_values('group', kind='mergesort')
        data['color'] = to_rgba(data['color'], data['alpha'])
        if self.type == 'open':
            data['facecolor'] = 'none'
        else:
            data['facecolor'] = data['color']
        if not constant:
            from matplotlib.collections import PathCollection
            idx1: list[int] = []
            idx2: list[int] = []
            for _, df in data.groupby('group'):
                idx1.extend(df.index[:-1].to_list())
                idx2.extend(df.index[1:].to_list())
            d = {'zorder': params['zorder'], 'rasterized': params['raster'], 'edgecolor': data.loc[idx1, 'color'], 'facecolor': data.loc[idx1, 'facecolor'], 'linewidth': data.loc[idx1, 'size'], 'linestyle': data.loc[idx1, 'linetype']}
            x1 = data.loc[idx1, 'x'].to_numpy()
            y1 = data.loc[idx1, 'y'].to_numpy()
            x2 = data.loc[idx2, 'x'].to_numpy()
            y2 = data.loc[idx2, 'y'].to_numpy()
            if first:
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                coll = PathCollection(paths, **d)
                ax.add_collection(coll)
            if last:
                x1, y1, x2, y2 = (x2, y2, x1, y1)
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                coll = PathCollection(paths, **d)
                ax.add_collection(coll)
        else:
            from matplotlib.patches import PathPatch
            d = {'zorder': params['zorder'], 'rasterized': params['raster'], 'edgecolor': data['color'].iloc[0], 'facecolor': data['facecolor'].iloc[0], 'linewidth': data['size'].iloc[0], 'linestyle': data['linetype'].iloc[0], 'joinstyle': 'round', 'capstyle': 'butt'}
            if first:
                x1, x2 = data['x'].iloc[0:2]
                y1, y2 = data['y'].iloc[0:2]
                x1, y1, x2, y2 = (np.array([i]) for i in (x1, y1, x2, y2))
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                patch = PathPatch(paths[0], **d)
                ax.add_artist(patch)
            if last:
                x1, x2 = data['x'].iloc[-2:]
                y1, y2 = data['y'].iloc[-2:]
                x1, y1, x2, y2 = (x2, y2, x1, y1)
                x1, y1, x2, y2 = (np.array([i]) for i in (x1, y1, x2, y2))
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                patch = PathPatch(paths[0], **d)
                ax.add_artist(patch)

    def get_paths(self, x1: npt.ArrayLike, y1: npt.ArrayLike, x2: npt.ArrayLike, y2: npt.ArrayLike, panel_params: panel_view, coord: coord, ax: Axes) -> list[Path]:
        """
        Compute paths that create the arrow heads

        Parameters
        ----------
        x1, y1, x2, y2 : array_like
            List of points that define the tails of the arrows.
            The arrow heads will be at x1, y1. If you need them
            at x2, y2 reverse the input.
        panel_params : panel_view
            The scale information as may be required by the
            axes. At this point, that information is about
            ranges, ticks and labels. Attributes are of interest
            to the geom are:

            ```python
            "panel_params.x.range"  # tuple
            "panel_params.y.range"  # tuple
            ```
        coord : coord
            Coordinate (e.g. coord_cartesian) system of the geom.
        ax : axes
            Axes on which to plot.

        Returns
        -------
        out : list of Path
            Paths that create arrow heads
        """
        from matplotlib.path import Path
        dummy = (0, 0)
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.STOP]
        width, height = _axes_get_size_inches(ax)
        ranges = coord.range(panel_params)
        width_ = np.ptp(ranges.x)
        height_ = np.ptp(ranges.y)
        lx = self.length * width_ / width
        ly = self.length * height_ / height
        a = self.angle * np.pi / 180
        xdiff, ydiff = (x2 - x1, y2 - y1)
        rotations = np.arctan2(ydiff / ly, xdiff / lx)
        v1x = x1 + lx * np.cos(rotations + a)
        v1y = y1 + ly * np.sin(rotations + a)
        v2x = x1 + lx * np.cos(rotations - a)
        v2y = y1 + ly * np.sin(rotations - a)
        paths = []
        for t in zip(v1x, v1y, x1, y1, v2x, v2y):
            verts = [t[:2], t[2:4], t[4:], dummy]
            paths.append(Path(verts, codes))
        return paths