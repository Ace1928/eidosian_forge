from functools import reduce
import numpy as np
import param
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .layout import AdjointLayout, Composable, Layout, Layoutable
from .ndmapping import UniformNdMapping
from .util import dimensioned_streams, sanitize_identifier, unique_array
class CompositeOverlay(ViewableElement, Composable):
    """
    CompositeOverlay provides a common baseclass for Overlay classes.
    """
    _deep_indexable = True

    def hist(self, dimension=None, num_bins=20, bin_range=None, adjoin=True, index=None, show_legend=False, **kwargs):
        """Computes and adjoins histogram along specified dimension(s).

        Defaults to first value dimension if present otherwise falls
        back to first key dimension.

        Args:
            dimension: Dimension(s) to compute histogram on,
                Falls back the plot dimensions by default.
            num_bins (int, optional): Number of bins
            bin_range (tuple optional): Lower and upper bounds of bins
            adjoin (bool, optional): Whether to adjoin histogram
            index (int, optional): Index of layer to apply hist to
            show_legend (bool, optional): Show legend in histogram
                (don't show legend by default).

        Returns:
            AdjointLayout of element and histogram or just the
            histogram
        """
        main_layer_int_index = getattr(self, 'main_layer', None) or 0
        if index is not None:
            valid_ind = isinstance(index, int) and 0 <= index < len(self)
            valid_label = index in [el.label for el in self]
            if not any([valid_ind, valid_label]):
                raise TypeError('Please supply a suitable index or label for the histogram data')
            if valid_ind:
                main_layer_int_index = index
            if valid_label:
                main_layer_int_index = self.keys().index(index)
        if dimension is None:
            dimension = [dim.name for dim in self.values()[main_layer_int_index].kdims]
        hists_per_dim = {dim: dict([(elem_key, elem.hist(adjoin=False, dimension=dim, bin_range=bin_range, num_bins=num_bins, **kwargs)) for i, (elem_key, elem) in enumerate(self.items()) if index is None or getattr(elem, 'label', None) == index or index == i]) for dim in dimension}
        hists_overlay_per_dim = {dim: self.clone(hists).opts(show_legend=show_legend) for dim, hists in hists_per_dim.items()}
        if adjoin:
            layout = self
            for dim in reversed(self.values()[main_layer_int_index].kdims):
                if dim.name in hists_overlay_per_dim:
                    layout = layout << hists_overlay_per_dim[dim.name]
            layout.main_layer = main_layer_int_index
        elif len(dimension) > 1:
            layout = Layout(list(hists_overlay_per_dim.values()))
        else:
            layout = hists_overlay_per_dim[0]
        return layout

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
                Whether to return the expanded values, behavior depends
                on the type of data:
                  * Columnar: If false returns unique values
                  * Geometry: If false returns scalar values per geometry
                  * Gridded: If false returns 1D coordinates
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        values = []
        found = False
        for el in self:
            if dimension in el.dimensions(label=True):
                values.append(el.dimension_values(dimension))
                found = True
        if not found:
            return super().dimension_values(dimension, expanded, flat)
        values = [v for v in values if v is not None and len(v)]
        if not values:
            return np.array()
        vals = np.concatenate(values)
        return vals if expanded else unique_array(vals)