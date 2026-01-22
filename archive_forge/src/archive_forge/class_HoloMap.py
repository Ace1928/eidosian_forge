import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
class HoloMap(Layoutable, UniformNdMapping, Overlayable):
    """
    A HoloMap is an n-dimensional mapping of viewable elements or
    overlays. Each item in a HoloMap has an tuple key defining the
    values along each of the declared key dimensions, defining the
    discretely sampled space of values.

    The visual representation of a HoloMap consists of the viewable
    objects inside the HoloMap which can be explored by varying one
    or more widgets mapping onto the key dimensions of the HoloMap.
    """
    data_type = (ViewableElement, NdMapping, Layout)

    def __init__(self, initial_items=None, kdims=None, group=None, label=None, **params):
        super().__init__(initial_items, kdims, group, label, **params)

    @property
    def opts(self):
        return Opts(self, mode='holomap')

    def overlay(self, dimensions=None, **kwargs):
        """Group by supplied dimension(s) and overlay each group

        Groups data by supplied dimension(s) overlaying the groups
        along the dimension(s).

        Args:
            dimensions: Dimension(s) of dimensions to group by

        Returns:
            NdOverlay object(s) with supplied dimensions
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return NdOverlay(self, **kwargs).reindex(dimensions)
        else:
            dims = [d for d in self.kdims if d not in dimensions]
            return self.groupby(dims, group_type=NdOverlay, **kwargs)

    def grid(self, dimensions=None, **kwargs):
        """Group by supplied dimension(s) and lay out groups in grid

        Groups data by supplied dimension(s) laying the groups along
        the dimension(s) out in a GridSpace.

        Args:
        dimensions: Dimension/str or list
            Dimension or list of dimensions to group by

        Returns:
            GridSpace with supplied dimensions
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return GridSpace(self, **kwargs).reindex(dimensions)
        return self.groupby(dimensions, container_type=GridSpace, **kwargs)

    def layout(self, dimensions=None, **kwargs):
        """Group by supplied dimension(s) and lay out groups

        Groups data by supplied dimension(s) laying the groups along
        the dimension(s) out in a NdLayout.

        Args:
            dimensions: Dimension(s) to group by

        Returns:
            NdLayout with supplied dimensions
        """
        dimensions = self._valid_dimensions(dimensions)
        if len(dimensions) == self.ndims:
            with item_check(False):
                return NdLayout(self, **kwargs).reindex(dimensions)
        return self.groupby(dimensions, container_type=NdLayout, **kwargs)

    def options(self, *args, **kwargs):
        """Applies simplified option definition returning a new object

        Applies options defined in a flat format to the objects
        returned by the DynamicMap. If the options are to be set
        directly on the objects in the HoloMap a simple format may be
        used, e.g.:

            obj.options(cmap='viridis', show_title=False)

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.options('Image', cmap='viridis', show_title=False)

        or using:

            obj.options({'Image': dict(cmap='viridis', show_title=False)})

        Args:
            *args: Sets of options to apply to object
                Supports a number of formats including lists of Options
                objects, a type[.group][.label] followed by a set of
                keyword options to apply and a dictionary indexed by
                type[.group][.label] specs.
            backend (optional): Backend to apply options to
                Defaults to current selected backend
            clone (bool, optional): Whether to clone object
                Options can be applied inplace with clone=False
            **kwargs: Keywords of options
                Set of options to apply to the object

        Returns:
            Returns the cloned object with the options applied
        """
        data = dict([(k, v.options(*args, **kwargs)) for k, v in self.data.items()])
        return self.clone(data)

    def _split_overlays(self):
        """Splits overlays inside the HoloMap into list of HoloMaps"""
        if not issubclass(self.type, CompositeOverlay):
            return (None, self.clone())
        item_maps = {}
        for k, overlay in self.data.items():
            for key, el in overlay.items():
                if key not in item_maps:
                    item_maps[key] = [(k, el)]
                else:
                    item_maps[key].append((k, el))
        maps, keys = ([], [])
        for k, layermap in item_maps.items():
            maps.append(self.clone(layermap))
            keys.append(k)
        return (keys, maps)

    def _dimension_keys(self):
        """
        Helper for __mul__ that returns the list of keys together with
        the dimension labels.
        """
        return [tuple(zip([d.name for d in self.kdims], [k] if self.ndims == 1 else k)) for k in self.keys()]

    def _dynamic_mul(self, dimensions, other, keys):
        """
        Implements dynamic version of overlaying operation overlaying
        DynamicMaps and HoloMaps where the key dimensions of one is
        a strict superset of the other.
        """
        if not isinstance(self, DynamicMap) or not isinstance(other, DynamicMap):
            keys = sorted(((d, v) for k in keys for d, v in k))
            grouped = {g: [v for _, v in group] for g, group in groupby(keys, lambda x: x[0])}
            dimensions = [d.clone(values=grouped[d.name]) for d in dimensions]
            map_obj = None
        map_obj = self if isinstance(self, DynamicMap) else other
        if isinstance(self, DynamicMap) and isinstance(other, DynamicMap):
            self_streams = util.dimensioned_streams(self)
            other_streams = util.dimensioned_streams(other)
            streams = list(util.unique_iterator(self_streams + other_streams))
        else:
            streams = map_obj.streams

        def dynamic_mul(*key, **kwargs):
            key_map = {d.name: k for d, k in zip(dimensions, key)}
            layers = []
            try:
                self_el = self.select(HoloMap, **key_map) if self.kdims else self[()]
                layers.append(self_el)
            except KeyError:
                pass
            try:
                other_el = other.select(HoloMap, **key_map) if other.kdims else other[()]
                layers.append(other_el)
            except KeyError:
                pass
            return Overlay(layers)
        callback = Callable(dynamic_mul, inputs=[self, other])
        callback._is_overlay = True
        if map_obj:
            return map_obj.clone(callback=callback, shared_data=False, kdims=dimensions, streams=streams)
        else:
            return DynamicMap(callback=callback, kdims=dimensions, streams=streams)

    def __mul__(self, other, reverse=False):
        """Overlays items in the object with another object

        The mul (*) operator implements overlaying of different
        objects.  This method tries to intelligently overlay mappings
        with differing keys. If the UniformNdMapping is mulled with a
        simple ViewableElement each element in the UniformNdMapping is
        overlaid with the ViewableElement. If the element the
        UniformNdMapping is mulled with is another UniformNdMapping it
        will try to match up the dimensions, making sure that items
        with completely different dimensions aren't overlaid.
        """
        if isinstance(other, HoloMap):
            self_set = {d.name for d in self.kdims}
            other_set = {d.name for d in other.kdims}
            self_in_other = self_set.issubset(other_set)
            other_in_self = other_set.issubset(self_set)
            dims = [other.kdims, self.kdims] if self_in_other else [self.kdims, other.kdims]
            dimensions = util.merge_dimensions(dims)
            if self_in_other and other_in_self:
                keys = self._dimension_keys() + other._dimension_keys()
                super_keys = util.unique_iterator(keys)
            elif self_in_other:
                dimensions = other.kdims
                super_keys = other._dimension_keys()
            elif other_in_self:
                super_keys = self._dimension_keys()
            else:
                raise Exception('One set of keys needs to be a strict subset of the other.')
            if isinstance(self, DynamicMap) or isinstance(other, DynamicMap):
                return self._dynamic_mul(dimensions, other, super_keys)
            items = []
            for dim_keys in super_keys:
                self_key = tuple((k for p, k in sorted([(self.get_dimension_index(dim), v) for dim, v in dim_keys if dim in self.kdims])))
                other_key = tuple((k for p, k in sorted([(other.get_dimension_index(dim), v) for dim, v in dim_keys if dim in other.kdims])))
                new_key = self_key if other_in_self else other_key
                if self_key in self and other_key in other:
                    if reverse:
                        value = other[other_key] * self[self_key]
                    else:
                        value = self[self_key] * other[other_key]
                    items.append((new_key, value))
                elif self_key in self:
                    items.append((new_key, Overlay([self[self_key]])))
                else:
                    items.append((new_key, Overlay([other[other_key]])))
            return self.clone(items, kdims=dimensions, label=self._label, group=self._group)
        elif isinstance(other, self.data_type) and (not isinstance(other, Layout)):
            if isinstance(self, DynamicMap):

                def dynamic_mul(*args, **kwargs):
                    element = self[args]
                    if reverse:
                        return other * element
                    else:
                        return element * other
                callback = Callable(dynamic_mul, inputs=[self, other])
                callback._is_overlay = True
                return self.clone(shared_data=False, callback=callback, streams=util.dimensioned_streams(self))
            items = [(k, other * v) if reverse else (k, v * other) for k, v in self.data.items()]
            return self.clone(items, label=self._label, group=self._group)
        else:
            return NotImplemented

    def __lshift__(self, other):
        """Adjoin another object to this one returning an AdjointLayout"""
        if isinstance(other, (ViewableElement, UniformNdMapping, Empty)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data + [self])
        else:
            raise TypeError(f'Cannot append {type(other).__name__} to a AdjointLayout')

    def collate(self, merge_type=None, drop=None, drop_constant=False):
        """Collate allows reordering nested containers

        Collation allows collapsing nested mapping types by merging
        their dimensions. In simple terms in merges nested containers
        into a single merged type.

        In the simple case a HoloMap containing other HoloMaps can
        easily be joined in this way. However collation is
        particularly useful when the objects being joined are deeply
        nested, e.g. you want to join multiple Layouts recorded at
        different times, collation will return one Layout containing
        HoloMaps indexed by Time. Changing the merge_type will allow
        merging the outer Dimension into any other UniformNdMapping
        type.

        Args:
            merge_type: Type of the object to merge with
            drop: List of dimensions to drop
            drop_constant: Drop constant dimensions automatically

        Returns:
            Collated Layout or HoloMap
        """
        if drop is None:
            drop = []
        from .element import Collator
        merge_type = merge_type if merge_type else self.__class__
        return Collator(self, merge_type=merge_type, drop=drop, drop_constant=drop_constant)()

    def decollate(self):
        """Packs HoloMap of DynamicMaps into a single DynamicMap that returns an
        HoloMap

        Decollation allows packing a HoloMap of DynamicMaps into a single DynamicMap
        that returns an HoloMap of simple (non-dynamic) elements. All nested streams
        are lifted to the resulting DynamicMap, and are available in the `streams`
        property.  The `callback` property of the resulting DynamicMap is a pure,
        stateless function of the stream values. To avoid stream parameter name
        conflicts, the resulting DynamicMap is configured with
        positional_stream_args=True, and the callback function accepts stream values
        as positional dict arguments.

        Returns:
            DynamicMap that returns an HoloMap
        """
        from .decollate import decollate
        return decollate(self)

    def relabel(self, label=None, group=None, depth=1):
        """Clone object and apply new group and/or label.

        Applies relabeling to children up to the supplied depth.

        Args:
            label (str, optional): New label to apply to returned object
            group (str, optional): New group to apply to returned object
            depth (int, optional): Depth to which relabel will be applied
                If applied to container allows applying relabeling to
                contained objects up to the specified depth

        Returns:
            Returns relabelled object
        """
        return super().relabel(label=label, group=group, depth=depth)

    def hist(self, dimension=None, num_bins=20, bin_range=None, adjoin=True, individually=True, **kwargs):
        """Computes and adjoins histogram along specified dimension(s).

        Defaults to first value dimension if present otherwise falls
        back to first key dimension.

        Args:
            dimension: Dimension(s) to compute histogram on
            num_bins (int, optional): Number of bins
            bin_range (tuple optional): Lower and upper bounds of bins
            adjoin (bool, optional): Whether to adjoin histogram

        Returns:
            AdjointLayout of HoloMap and histograms or just the
            histograms
        """
        if dimension is not None and (not isinstance(dimension, list)):
            dimension = [dimension]
        histmaps = [self.clone(shared_data=False) for _ in dimension or [None]]
        if individually:
            map_range = None
        else:
            if dimension is None:
                raise Exception('Please supply the dimension to compute a histogram for.')
            map_range = self.range(kwargs['dimension'])
        bin_range = map_range if bin_range is None else bin_range
        style_prefix = 'Custom[<' + self.name + '>]_'
        if issubclass(self.type, (NdOverlay, Overlay)) and 'index' not in kwargs:
            kwargs['index'] = 0
        for k, v in self.data.items():
            hists = v.hist(adjoin=False, dimension=dimension, bin_range=bin_range, num_bins=num_bins, style_prefix=style_prefix, **kwargs)
            if isinstance(hists, Layout):
                for i, hist in enumerate(hists):
                    histmaps[i][k] = hist
            else:
                histmaps[0][k] = hists
        if adjoin:
            layout = self
            for hist in histmaps:
                layout = layout << hist
            if issubclass(self.type, (NdOverlay, Overlay)):
                layout.main_layer = kwargs['index']
            return layout
        elif len(histmaps) > 1:
            return Layout(histmaps)
        else:
            return histmaps[0]