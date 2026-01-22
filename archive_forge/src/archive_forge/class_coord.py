from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
class coord:
    """
    Base class for all coordinate systems
    """
    is_linear = False
    params: dict[str, Any]

    def __radd__(self, plot: ggplot) -> ggplot:
        """
        Add coordinates to ggplot object
        """
        plot.coordinates = copy(self)
        return plot

    def setup_data(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Allow the coordinate system to manipulate the layer data

        Parameters
        ----------
        data :
            Data for alls Layer

        Returns
        -------
        :
            Modified layer data
        """
        return data

    def setup_params(self, data: list[pd.DataFrame]):
        """
        Create additional parameters

        A coordinate system may need to create parameters
        depending on the *original* data that the layers get.

        Parameters
        ----------
        data :
            Data for each layer before it is manipulated in
            any way.
        """
        self.params = {}

    def setup_layout(self, layout: pd.DataFrame) -> pd.DataFrame:
        """
        Allow the coordinate system alter the layout dataframe

        Parameters
        ----------
        layout :
            Dataframe in which data is assigned to panels and scales

        Returns
        -------
        :
            layout dataframe altered to according to the requirements
            of the coordinate system.

        Notes
        -----
        The input dataframe may be changed.
        """
        return layout

    def aspect(self, panel_params: panel_view) -> float | None:
        """
        Return desired aspect ratio for the plot

        If not overridden by the subclass, this method
        returns `None`, which means that the coordinate
        system does not influence the aspect ratio.
        """
        return None

    def labels(self, cur_labels: labels_view) -> labels_view:
        """
        Modify labels

        Parameters
        ----------
        cur_labels :
            Current labels. The coord can modify them as necessary.

        Returns
        -------
        :
            Modified labels. Same object as the input.
        """
        return cur_labels

    def transform(self, data: pd.DataFrame, panel_params: panel_view, munch: bool=False) -> pd.DataFrame:
        """
        Transform data before it is plotted

        This is used to "transform the coordinate axes".
        Subclasses should override this method
        """
        return data

    def setup_panel_params(self, scale_x: scale, scale_y: scale) -> panel_view:
        """
        Compute the range and break information for the panel
        """
        msg = 'The coordinate should implement this method.'
        raise NotImplementedError(msg)

    def range(self, panel_params: panel_view) -> panel_ranges:
        """
        Return the range along the dimensions of the coordinate system
        """
        return panel_ranges(x=panel_params.x.range, y=panel_params.y.range)

    def backtransform_range(self, panel_params: panel_view) -> panel_ranges:
        """
        Backtransform the panel range in panel_params to data coordinates

        Coordinate systems that do any transformations should override
        this method. e.g. coord_trans has to override this method.
        """
        return self.range(panel_params)

    def distance(self, x: FloatSeries, y: FloatSeries, panel_params: panel_view) -> npt.NDArray[Any]:
        msg = 'The coordinate should implement this method.'
        raise NotImplementedError(msg)

    def munch(self, data: pd.DataFrame, panel_params: panel_view) -> pd.DataFrame:
        ranges = self.backtransform_range(panel_params)
        x_neginf = np.isneginf(data['x'])
        x_posinf = np.isposinf(data['x'])
        y_neginf = np.isneginf(data['y'])
        y_posinf = np.isposinf(data['y'])
        if x_neginf.any():
            data.loc[x_neginf, 'x'] = ranges.x[0]
        if x_posinf.any():
            data.loc[x_posinf, 'x'] = ranges.x[1]
        if y_neginf.any():
            data.loc[y_neginf, 'y'] = ranges.y[0]
        if y_posinf.any():
            data.loc[y_posinf, 'y'] = ranges.y[1]
        dist = self.distance(data['x'], data['y'], panel_params)
        bool_idx = data['group'].to_numpy()[1:] != data['group'].to_numpy()[:-1]
        dist[bool_idx] = np.nan
        munched = munch_data(data, dist)
        return munched