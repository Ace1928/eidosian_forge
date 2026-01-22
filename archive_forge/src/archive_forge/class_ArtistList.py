from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
class ArtistList(Sequence):
    """
        A sublist of Axes children based on their type.

        The type-specific children sublists were made immutable in Matplotlib
        3.7.  In the future these artist lists may be replaced by tuples. Use
        as if this is a tuple already.
        """

    def __init__(self, axes, prop_name, valid_types=None, invalid_types=None):
        """
            Parameters
            ----------
            axes : `~matplotlib.axes.Axes`
                The Axes from which this sublist will pull the children
                Artists.
            prop_name : str
                The property name used to access this sublist from the Axes;
                used to generate deprecation warnings.
            valid_types : list of type, optional
                A list of types that determine which children will be returned
                by this sublist. If specified, then the Artists in the sublist
                must be instances of any of these types. If unspecified, then
                any type of Artist is valid (unless limited by
                *invalid_types*.)
            invalid_types : tuple, optional
                A list of types that determine which children will *not* be
                returned by this sublist. If specified, then Artists in the
                sublist will never be an instance of these types. Otherwise, no
                types will be excluded.
            """
        self._axes = axes
        self._prop_name = prop_name
        self._type_check = lambda artist: (not valid_types or isinstance(artist, valid_types)) and (not invalid_types or not isinstance(artist, invalid_types))

    def __repr__(self):
        return f'<Axes.ArtistList of {len(self)} {self._prop_name}>'

    def __len__(self):
        return sum((self._type_check(artist) for artist in self._axes._children))

    def __iter__(self):
        for artist in list(self._axes._children):
            if self._type_check(artist):
                yield artist

    def __getitem__(self, key):
        return [artist for artist in self._axes._children if self._type_check(artist)][key]

    def __add__(self, other):
        if isinstance(other, (list, _AxesBase.ArtistList)):
            return [*self, *other]
        if isinstance(other, (tuple, _AxesBase.ArtistList)):
            return (*self, *other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return other + list(self)
        if isinstance(other, tuple):
            return other + tuple(self)
        return NotImplemented