import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
class Dimensioned(LabelledData):
    """
    Dimensioned is a base class that allows the data contents of a
    class to be associated with dimensions. The contents associated
    with dimensions may be partitioned into one of three types

    * key dimensions: These are the dimensions that can be indexed via
                      the __getitem__ method. Dimension objects
                      supporting key dimensions must support indexing
                      over these dimensions and may also support
                      slicing. This list ordering of dimensions
                      describes the positional components of each
                      multi-dimensional indexing operation.

                      For instance, if the key dimension names are
                      'weight' followed by 'height' for Dimensioned
                      object 'obj', then obj[80,175] indexes a weight
                      of 80 and height of 175.

                      Accessed using either kdims.

    * value dimensions: These dimensions correspond to any data held
                        on the Dimensioned object not in the key
                        dimensions. Indexing by value dimension is
                        supported by dimension name (when there are
                        multiple possible value dimensions); no
                        slicing semantics is supported and all the
                        data associated with that dimension will be
                        returned at once. Note that it is not possible
                        to mix value dimensions and deep dimensions.

                        Accessed using either vdims.

    * deep dimensions: These are dynamically computed dimensions that
                       belong to other Dimensioned objects that are
                       nested in the data. Objects that support this
                       should enable the _deep_indexable flag. Note
                       that it is not possible to mix value dimensions
                       and deep dimensions.

                       Accessed using either ddims.

    Dimensioned class support generalized methods for finding the
    range and type of values along a particular Dimension. The range
    method relies on the appropriate implementation of the
    dimension_values methods on subclasses.

    The index of an arbitrary dimension is its positional index in the
    list of all dimensions, starting with the key dimensions, followed
    by the value dimensions and ending with the deep dimensions.
    """
    cdims = param.Dict(default={}, doc='\n       The constant dimensions defined as a dictionary of Dimension:value\n       pairs providing additional dimension information about the object.\n\n       Aliased with constant_dimensions.')
    kdims = param.List(bounds=(0, None), constant=True, doc='\n       The key dimensions defined as list of dimensions that may be\n       used in indexing (and potential slicing) semantics. The order\n       of the dimensions listed here determines the semantics of each\n       component of a multi-dimensional indexing operation.\n\n       Aliased with key_dimensions.')
    vdims = param.List(bounds=(0, None), constant=True, doc='\n       The value dimensions defined as the list of dimensions used to\n       describe the components of the data. If multiple value\n       dimensions are supplied, a particular value dimension may be\n       indexed by name after the key dimensions.\n\n       Aliased with value_dimensions.')
    group = param.String(default='Dimensioned', constant=True, doc='\n       A string describing the data wrapped by the object.')
    __abstract = True
    _dim_groups = ['kdims', 'vdims', 'cdims', 'ddims']
    _dim_aliases = dict(key_dimensions='kdims', value_dimensions='vdims', constant_dimensions='cdims', deep_dimensions='ddims')

    def __init__(self, data, kdims=None, vdims=None, **params):
        params.update(process_dimensions(kdims, vdims))
        if 'cdims' in params:
            params['cdims'] = {d if isinstance(d, Dimension) else Dimension(d): val for d, val in params['cdims'].items()}
        super().__init__(data, **params)
        self.ndims = len(self.kdims)
        cdims = [(d.name, val) for d, val in self.cdims.items()]
        self._cached_constants = dict(cdims)
        self._settings = None

    @property
    def apply(self):
        return Apply(self)

    @property
    def opts(self):
        return Opts(self)

    @property
    def redim(self):
        return Redim(self)

    def _valid_dimensions(self, dimensions):
        """Validates key dimension input

        Returns kdims if no dimensions are specified"""
        if dimensions is None:
            dimensions = self.kdims
        elif not isinstance(dimensions, list):
            dimensions = [dimensions]
        valid_dimensions = []
        for dim in dimensions:
            if isinstance(dim, Dimension):
                dim = dim.name
            if dim not in self.kdims:
                raise Exception(f'Supplied dimensions {dim} not found.')
            valid_dimensions.append(dim)
        return valid_dimensions

    @property
    def ddims(self):
        """The list of deep dimensions"""
        if self._deep_indexable and self:
            return self.values()[0].dimensions()
        else:
            return []

    def dimensions(self, selection='all', label=False):
        """Lists the available dimensions on the object

        Provides convenient access to Dimensions on nested Dimensioned
        objects. Dimensions can be selected by their type, i.e. 'key'
        or 'value' dimensions. By default 'all' dimensions are
        returned.

        Args:
            selection: Type of dimensions to return
                The type of dimension, i.e. one of 'key', 'value',
                'constant' or 'all'.
            label: Whether to return the name, label or Dimension
                Whether to return the Dimension objects (False),
                the Dimension names (True/'name') or labels ('label').

        Returns:
            List of Dimension objects or their names or labels
        """
        if label in ['name', True]:
            label = 'short'
        elif label == 'label':
            label = 'long'
        elif label:
            raise ValueError("label needs to be one of True, False, 'name' or 'label'")
        lambdas = {'k': (lambda x: x.kdims, {'full_breadth': False}), 'v': (lambda x: x.vdims, {}), 'c': (lambda x: x.cdims, {})}
        aliases = {'key': 'k', 'value': 'v', 'constant': 'c'}
        if selection in ['all', 'ranges']:
            groups = [d for d in self._dim_groups if d != 'cdims']
            dims = [dim for group in groups for dim in getattr(self, group)]
        elif isinstance(selection, list):
            dims = [dim for group in selection for dim in getattr(self, f'{aliases.get(group)}dims')]
        elif aliases.get(selection) in lambdas:
            selection = aliases.get(selection, selection)
            lmbd, kwargs = lambdas[selection]
            key_traversal = self.traverse(lmbd, **kwargs)
            dims = [dim for keydims in key_traversal for dim in keydims]
        else:
            raise KeyError("Invalid selection %r, valid selections include'all', 'value' and 'key' dimensions" % repr(selection))
        return [(dim.label if label == 'long' else dim.name) if label else dim for dim in dims]

    def get_dimension(self, dimension, default=None, strict=False):
        """Get a Dimension object by name or index.

        Args:
            dimension: Dimension to look up by name or integer index
            default (optional): Value returned if Dimension not found
            strict (bool, optional): Raise a KeyError if not found

        Returns:
            Dimension object for the requested dimension or default
        """
        if dimension is not None and (not isinstance(dimension, (int, str, Dimension))):
            raise TypeError('Dimension lookup supports int, string, and Dimension instances, cannot lookup Dimensions using %s type.' % type(dimension).__name__)
        all_dims = self.dimensions()
        if isinstance(dimension, int):
            if 0 <= dimension < len(all_dims):
                return all_dims[dimension]
            elif strict:
                raise KeyError(f'Dimension {dimension!r} not found')
            else:
                return default
        if isinstance(dimension, Dimension):
            dims = [d for d in all_dims if dimension == d]
            if strict and (not dims):
                raise KeyError(f'{dimension!r} not found.')
            elif dims:
                return dims[0]
            else:
                return None
        else:
            dimension = dimension_name(dimension)
            name_map = {dim.spec: dim for dim in all_dims}
            name_map.update({dim.name: dim for dim in all_dims})
            name_map.update({dim.label: dim for dim in all_dims})
            name_map.update({util.dimension_sanitizer(dim.name): dim for dim in all_dims})
            if strict and dimension not in name_map:
                raise KeyError(f'Dimension {dimension!r} not found.')
            else:
                return name_map.get(dimension, default)

    def get_dimension_index(self, dimension):
        """Get the index of the requested dimension.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Integer index of the requested dimension
        """
        if isinstance(dimension, int):
            if dimension < self.ndims + len(self.vdims) or dimension < len(self.dimensions()):
                return dimension
            else:
                return IndexError('Dimension index out of bounds')
        dim = dimension_name(dimension)
        try:
            dimensions = self.kdims + self.vdims
            return next((i for i, d in enumerate(dimensions) if d == dim))
        except StopIteration:
            raise Exception(f'Dimension {dim} not found in {self.__class__.__name__}.') from None

    def get_dimension_type(self, dim):
        """Get the type of the requested dimension.

        Type is determined by Dimension.type attribute or common
        type of the dimension values, otherwise None.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Declared type of values along the dimension
        """
        dim_obj = self.get_dimension(dim)
        if dim_obj and dim_obj.type is not None:
            return dim_obj.type
        dim_vals = [type(v) for v in self.dimension_values(dim)]
        if len(set(dim_vals)) == 1:
            return dim_vals[0]
        else:
            return None

    def __getitem__(self, key):
        """
        Multi-dimensional indexing semantics is determined by the list
        of key dimensions. For instance, the first indexing component
        will index the first key dimension.

        After the key dimensions are given, *either* a value dimension
        name may follow (if there are multiple value dimensions) *or*
        deep dimensions may then be listed (for applicable deep
        dimensions).
        """
        return self

    def select(self, selection_specs=None, **kwargs):
        """Applies selection by dimension name

        Applies a selection along the dimensions of the object using
        keyword arguments. The selection may be narrowed to certain
        objects using selection_specs. For container objects the
        selection will be applied to all children as well.

        Selections may select a specific value, slice or set of values:

        * value: Scalar values will select rows along with an exact
                 match, e.g.:

            ds.select(x=3)

        * slice: Slices may be declared as tuples of the upper and
                 lower bound, e.g.:

            ds.select(x=(0, 3))

        * values: A list of values may be selected using a list or
                  set, e.g.:

            ds.select(x=[0, 1, 2])

        Args:
            selection_specs: List of specs to match on
                A list of types, functions, or type[.group][.label]
                strings specifying which objects to apply the
                selection on.
            **selection: Dictionary declaring selections by dimension
                Selections can be scalar values, tuple ranges, lists
                of discrete values and boolean arrays

        Returns:
            Returns an Dimensioned object containing the selected data
            or a scalar if a single value was selected
        """
        if selection_specs is not None and (not isinstance(selection_specs, (list, tuple))):
            selection_specs = [selection_specs]
        vdims = self.vdims + ['value'] if self.vdims else []
        kdims = self.kdims
        local_kwargs = {k: v for k, v in kwargs.items() if k in kdims + vdims}
        if selection_specs is not None:
            if not isinstance(selection_specs, (list, tuple)):
                selection_specs = [selection_specs]
            matches = any((self.matches(spec) for spec in selection_specs))
        else:
            matches = True
        if local_kwargs and matches:
            ndims = self.ndims
            if any((d in self.vdims for d in kwargs)):
                ndims = len(self.kdims + self.vdims)
            select = [slice(None) for _ in range(ndims)]
            for dim, val in local_kwargs.items():
                if dim == 'value':
                    select += [val]
                else:
                    if isinstance(val, tuple):
                        val = slice(*val)
                    select[self.get_dimension_index(dim)] = val
            if self._deep_indexable:
                selection = self.get(tuple(select), None)
                if selection is None:
                    selection = self.clone(shared_data=False)
            else:
                selection = self[tuple(select)]
        else:
            selection = self
        if not isinstance(selection, Dimensioned):
            return selection
        elif type(selection) is not type(self) and isinstance(selection, Dimensioned):
            dimensions = selection.dimensions() + ['value']
            if any((kw in dimensions for kw in kwargs)):
                selection = selection.select(selection_specs=selection_specs, **kwargs)
        elif isinstance(selection, Dimensioned) and selection._deep_indexable:
            items = []
            for k, v in selection.items():
                dimensions = v.dimensions() + ['value']
                if any((kw in dimensions for kw in kwargs)):
                    items.append((k, v.select(selection_specs=selection_specs, **kwargs)))
                else:
                    items.append((k, v))
            selection = selection.clone(items)
        return selection

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
        val = self._cached_constants.get(dimension, None)
        if val:
            return np.array([val])
        else:
            raise Exception(f'Dimension {dimension} not found in {self.__class__.__name__}.')

    def range(self, dimension, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        dimension = self.get_dimension(dimension)
        if dimension is None or (not data_range and (not dimension_range)):
            return (None, None)
        elif all((util.isfinite(v) for v in dimension.range)) and dimension_range:
            return dimension.range
        elif data_range:
            if dimension in self.kdims + self.vdims:
                dim_vals = self.dimension_values(dimension.name)
                lower, upper = util.find_range(dim_vals)
            else:
                dname = dimension.name
                match_fn = lambda x: dname in x.kdims + x.vdims
                range_fn = lambda x: x.range(dname)
                ranges = self.traverse(range_fn, [match_fn])
                lower, upper = util.max_range(ranges)
        else:
            lower, upper = (np.nan, np.nan)
        if not dimension_range:
            return (lower, upper)
        return util.dimension_range(lower, upper, dimension.range, dimension.soft_range)

    def __repr__(self):
        return PrettyPrinter.pprint(self)

    def __str__(self):
        return repr(self)

    def options(self, *args, clone=True, **kwargs):
        """Applies simplified option definition returning a new object.

        Applies options on an object or nested group of objects in a
        flat format returning a new object with the options
        applied. If the options are to be set directly on the object a
        simple format may be used, e.g.:

            obj.options(cmap='viridis', show_title=False)

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            obj.options('Image', cmap='viridis', show_title=False)

        or using:

            obj.options({'Image': dict(cmap='viridis', show_title=False)})

        Identical to the .opts method but returns a clone of the object
        by default.

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
        backend = kwargs.get('backend', None)
        if not (args or kwargs):
            options = None
        elif args and isinstance(args[0], str):
            options = {args[0]: kwargs}
        elif args and isinstance(args[0], list):
            if kwargs:
                raise ValueError('Please specify a list of option objects, or kwargs, but not both')
            options = args[0]
        elif args and [k for k in kwargs.keys() if k != 'backend']:
            raise ValueError("Options must be defined in one of two formats. Either supply keywords defining the options for the current object, e.g. obj.options(cmap='viridis'), or explicitly define the type, e.g. obj.options({'Image': {'cmap': 'viridis'}}). Supplying both formats is not supported.")
        elif args and all((isinstance(el, dict) for el in args)):
            if len(args) > 1:
                self.param.warning('Only a single dictionary can be passed as a positional argument. Only processing the first dictionary')
            options = [Options(spec, **kws) for spec, kws in args[0].items()]
        elif args:
            options = list(args)
        elif kwargs:
            options = {type(self).__name__: kwargs}
        from ..util import opts
        if options is None:
            expanded_backends = [(backend, {})]
        elif isinstance(options, list):
            expanded_backends = opts._expand_by_backend(options, backend)
        else:
            expanded_backends = [(backend, opts._expand_options(options, backend))]
        obj = self
        for backend, expanded in expanded_backends:
            obj = obj.opts._dispatch_opts(expanded, backend=backend, clone=clone)
        return obj

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        Resolves the class hierarchy for the class rendering the
        object using any display hooks registered on Store.display
        hooks.  The output of all registered display_hooks is then
        combined and returned.
        """
        return Store.render(self)