from __future__ import annotations
import copy
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import (
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.datatree_.datatree.common import TreeAttrAccessMixin
from xarray.datatree_.datatree.formatting import datatree_repr
from xarray.datatree_.datatree.formatting_html import (
from xarray.datatree_.datatree.mapping import (
from xarray.datatree_.datatree.ops import (
from xarray.datatree_.datatree.render import RenderTree
class DataTree(NamedNode, MappedDatasetMethodsMixin, MappedDataWithCoords, DataTreeArithmeticMixin, TreeAttrAccessMixin, Generic[Tree], Mapping):
    """
    A tree-like hierarchical collection of xarray objects.

    Attempts to present an API like that of xarray.Dataset, but methods are wrapped to also update all the tree's child nodes.
    """
    _name: str | None
    _parent: DataTree | None
    _children: dict[str, DataTree]
    _attrs: dict[Hashable, Any] | None
    _cache: dict[str, Any]
    _coord_names: set[Hashable]
    _dims: dict[Hashable, int]
    _encoding: dict[Hashable, Any] | None
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index]
    _variables: dict[Hashable, Variable]
    __slots__ = ('_name', '_parent', '_children', '_attrs', '_cache', '_coord_names', '_dims', '_encoding', '_close', '_indexes', '_variables')

    def __init__(self, data: Dataset | DataArray | None=None, parent: DataTree | None=None, children: Mapping[str, DataTree] | None=None, name: str | None=None):
        """
        Create a single node of a DataTree.

        The node may optionally contain data in the form of data and coordinate variables, stored in the same way as
        data is stored in an xarray.Dataset.

        Parameters
        ----------
        data : Dataset, DataArray, or None, optional
            Data to store under the .ds attribute of this node. DataArrays will be promoted to Datasets.
            Default is None.
        parent : DataTree, optional
            Parent node to this node. Default is None.
        children : Mapping[str, DataTree], optional
            Any child nodes of this node. Default is None.
        name : str, optional
            Name for this node of the tree. Default is None.

        Returns
        -------
        DataTree

        See Also
        --------
        DataTree.from_dict
        """
        if children is None:
            children = {}
        ds = _coerce_to_dataset(data)
        _check_for_name_collisions(children, ds.variables)
        super().__init__(name=name)
        self._replace(inplace=True, variables=ds._variables, coord_names=ds._coord_names, dims=ds._dims, indexes=ds._indexes, attrs=ds._attrs, encoding=ds._encoding)
        self._close = ds._close
        self.children = children
        self.parent = parent

    @property
    def parent(self: DataTree) -> DataTree | None:
        """Parent of this node."""
        return self._parent

    @parent.setter
    def parent(self: DataTree, new_parent: DataTree) -> None:
        if new_parent and self.name is None:
            raise ValueError('Cannot set an unnamed node as a child of another node')
        self._set_parent(new_parent, self.name)

    @property
    def ds(self) -> DatasetView:
        """
        An immutable Dataset-like view onto the data in this node.

        For a mutable Dataset containing the same data as in this node, use `.to_dataset()` instead.

        See Also
        --------
        DataTree.to_dataset
        """
        return DatasetView._from_node(self)

    @ds.setter
    def ds(self, data: Dataset | DataArray | None=None) -> None:
        ds = _coerce_to_dataset(data)
        _check_for_name_collisions(self.children, ds.variables)
        self._replace(inplace=True, variables=ds._variables, coord_names=ds._coord_names, dims=ds._dims, indexes=ds._indexes, attrs=ds._attrs, encoding=ds._encoding)
        self._close = ds._close

    def _pre_attach(self: DataTree, parent: DataTree) -> None:
        """
        Method which superclass calls before setting parent, here used to prevent having two
        children with duplicate names (or a data variable with the same name as a child).
        """
        super()._pre_attach(parent)
        if self.name in list(parent.ds.variables):
            raise KeyError(f'parent {parent.name} already contains a data variable named {self.name}')

    def to_dataset(self) -> Dataset:
        """
        Return the data in this node as a new xarray.Dataset object.

        See Also
        --------
        DataTree.ds
        """
        return Dataset._construct_direct(self._variables, self._coord_names, self._dims, self._attrs, self._indexes, self._encoding, self._close)

    @property
    def has_data(self):
        """Whether or not there are any data variables in this node."""
        return len(self._variables) > 0

    @property
    def has_attrs(self) -> bool:
        """Whether or not there are any metadata attributes in this node."""
        return len(self.attrs.keys()) > 0

    @property
    def is_empty(self) -> bool:
        """False if node contains any data or attrs. Does not look at children."""
        return not (self.has_data or self.has_attrs)

    @property
    def is_hollow(self) -> bool:
        """True if only leaf nodes contain data."""
        return not any((node.has_data for node in self.subtree if not node.is_leaf))

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to node contents as dict of Variable objects.

        This dictionary is frozen to prevent mutation that could violate
        Dataset invariants. It contains all variable objects constituting this
        DataTree node, including both data variables and coordinates.
        """
        return Frozen(self._variables)

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """Dictionary of global attributes on this node object."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self) -> dict:
        """Dictionary of global encoding attributes on this node object."""
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `DataTree.sizes`, `Dataset.sizes`, and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(self._dims)

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `DataTree.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        return self.dims

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        yield from self._item_sources
        yield self.attrs

    @property
    def _item_sources(self) -> Iterable[Mapping[Any, Any]]:
        """Places to look-up items for key-completion"""
        yield self.data_vars
        yield HybridMappingProxy(keys=self._coord_names, mapping=self.coords)
        yield HybridMappingProxy(keys=self.dims, mapping=self)
        yield self.children

    def _ipython_key_completions_(self) -> list[str]:
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        items_on_this_node = self._item_sources
        full_file_like_paths_to_all_nodes_in_subtree = {node.path[1:]: node for node in self.subtree}
        all_item_sources = itertools.chain(items_on_this_node, [full_file_like_paths_to_all_nodes_in_subtree])
        items = {item for source in all_item_sources for item in source if isinstance(item, str)}
        return list(items)

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is either an array stored in the datatree or a child node, or neither.
        """
        return key in self.variables or key in self.children

    def __bool__(self) -> bool:
        return bool(self.ds.data_vars) or bool(self.children)

    def __iter__(self) -> Iterator[Hashable]:
        return itertools.chain(self.ds.data_vars, self.children)

    def __array__(self, dtype=None):
        raise TypeError('cannot directly convert a DataTree into a numpy array. Instead, create an xarray.DataArray first, either with indexing on the DataTree or by invoking the `to_array()` method.')

    def __repr__(self) -> str:
        return datatree_repr(self)

    def __str__(self) -> str:
        return datatree_repr(self)

    def _repr_html_(self):
        """Make html representation of datatree object"""
        if XR_OPTS['display_style'] == 'text':
            return f'<pre>{escape(repr(self))}</pre>'
        return datatree_repr_html(self)

    @classmethod
    def _construct_direct(cls, variables: dict[Any, Variable], coord_names: set[Hashable], dims: dict[Any, int] | None=None, attrs: dict | None=None, indexes: dict[Any, Index] | None=None, encoding: dict | None=None, name: str | None=None, parent: DataTree | None=None, children: dict[str, DataTree] | None=None, close: Callable[[], None] | None=None) -> DataTree:
        """Shortcut around __init__ for internal use when we want to skip costly validation."""
        if dims is None:
            dims = calculate_dimensions(variables)
        if indexes is None:
            indexes = {}
        if children is None:
            children = dict()
        obj: DataTree = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._close = close
        obj._encoding = encoding
        obj._name = name
        obj._children = children
        obj._parent = parent
        return obj

    def _replace(self: DataTree, variables: dict[Hashable, Variable] | None=None, coord_names: set[Hashable] | None=None, dims: dict[Any, int] | None=None, attrs: dict[Hashable, Any] | None | Default=_default, indexes: dict[Hashable, Index] | None=None, encoding: dict | None | Default=_default, name: str | None | Default=_default, parent: DataTree | None | Default=_default, children: dict[str, DataTree] | None=None, inplace: bool=False) -> DataTree:
        """
        Fastpath constructor for internal use.

        Returns an object with optionally replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        datatree. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not None:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            if name is not _default:
                self._name = name
            if parent is not _default:
                self._parent = parent
            if children is not None:
                self._children = children
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is None:
                indexes = self._indexes.copy()
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            if name is _default:
                name = self._name
            if parent is _default:
                parent = copy.copy(self._parent)
            if children is _default:
                children = copy.copy(self._children)
            obj = self._construct_direct(variables, coord_names, dims, attrs, indexes, encoding, name, parent, children)
        return obj

    def copy(self: DataTree, deep: bool=False) -> DataTree:
        """
        Returns a copy of this subtree.

        Copies this node and all child nodes.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new datatree is the same as in
        the original datatree.

        Parameters
        ----------
        deep : bool, default: False
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.

        Returns
        -------
        object : DataTree
            New object with dimensions, attributes, coordinates, name, encoding,
            and data of this node and all child nodes copied from original.

        See Also
        --------
        xarray.Dataset.copy
        pandas.DataFrame.copy
        """
        return self._copy_subtree(deep=deep)

    def _copy_subtree(self: DataTree, deep: bool=False, memo: dict[int, Any] | None=None) -> DataTree:
        """Copy entire subtree"""
        new_tree = self._copy_node(deep=deep)
        for node in self.descendants:
            path = node.relative_to(self)
            new_tree[path] = node._copy_node(deep=deep)
        return new_tree

    def _copy_node(self: DataTree, deep: bool=False) -> DataTree:
        """Copy just one node of a tree"""
        new_node: DataTree = DataTree()
        new_node.name = self.name
        new_node.ds = self.to_dataset().copy(deep=deep)
        return new_node

    def __copy__(self: DataTree) -> DataTree:
        return self._copy_subtree(deep=False)

    def __deepcopy__(self: DataTree, memo: dict[int, Any] | None=None) -> DataTree:
        return self._copy_subtree(deep=True, memo=memo)

    def get(self: DataTree, key: str, default: DataTree | DataArray | None=None) -> DataTree | DataArray | None:
        """
        Access child nodes, variables, or coordinates stored in this node.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node. Must lie in this immediate node (not elsewhere in the tree).
        default : DataTree | DataArray | None, optional
            A value to return if the specified key does not exist. Default return value is None.
        """
        if key in self.children:
            return self.children[key]
        elif key in self.ds:
            return self.ds[key]
        else:
            return default

    def __getitem__(self: DataTree, key: str) -> DataTree | DataArray:
        """
        Access child nodes, variables, or coordinates stored anywhere in this tree.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node, or unix-like path to variable / child within another node.

        Returns
        -------
        DataTree | DataArray
        """
        if utils.is_dict_like(key):
            raise NotImplementedError('Should this index over whole tree?')
        elif isinstance(key, str):
            path = NodePath(key)
            return self._get_item(path)
        elif utils.is_list_like(key):
            raise NotImplementedError('Selecting via tags is deprecated, and selecting multiple items should be implemented via .subset')
        else:
            raise ValueError(f'Invalid format for key: {key}')

    def _set(self, key: str, val: DataTree | CoercibleValue) -> None:
        """
        Set the child node or variable with the specified key to value.

        Counterpart to the public .get method, and also only works on the immediate node, not other nodes in the tree.
        """
        if isinstance(val, DataTree):
            new_node = val.copy(deep=False)
            new_node.name = key
            new_node.parent = self
        else:
            if not isinstance(val, (DataArray, Variable)):
                val = DataArray(val)
            self.update({key: val})

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Add either a child node or an array to the tree, at any position.

        Data can be added anywhere, and new nodes will be created to cross the path to the new location if necessary.

        If there is already a node at the given location, then if value is a Node class or Dataset it will overwrite the
        data already present at that node, and if value is a single array, it will be merged with it.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError
        elif isinstance(key, str):
            path = NodePath(key)
            return self._set_item(path, value, new_nodes_along_path=True)
        else:
            raise ValueError('Invalid format for key')

    @overload
    def update(self, other: Dataset) -> None:
        ...

    @overload
    def update(self, other: Mapping[Hashable, DataArray | Variable]) -> None:
        ...

    @overload
    def update(self, other: Mapping[str, DataTree | DataArray | Variable]) -> None:
        ...

    def update(self, other: Dataset | Mapping[Hashable, DataArray | Variable] | Mapping[str, DataTree | DataArray | Variable]) -> None:
        """
        Update this node's children and / or variables.

        Just like `dict.update` this is an in-place operation.
        """
        new_children: dict[str, DataTree] = {}
        new_variables = {}
        for k, v in other.items():
            if isinstance(v, DataTree):
                new_child: DataTree = v.copy()
                new_child.name = str(k)
                new_children[str(k)] = new_child
            elif isinstance(v, (DataArray, Variable)):
                new_variables[k] = v
            else:
                raise TypeError(f'Type {type(v)} cannot be assigned to a DataTree')
        vars_merge_result = dataset_update_method(self.to_dataset(), new_variables)
        merged_children = {**self.children, **new_children}
        self._replace(inplace=True, children=merged_children, **vars_merge_result._asdict())

    def assign(self, items: Mapping[Any, Any] | None=None, **items_kwargs: Any) -> DataTree:
        """
        Assign new data variables or child nodes to a DataTree, returning a new object
        with all the original items in addition to the new ones.

        Parameters
        ----------
        items : mapping of hashable to Any
            Mapping from variable or child node names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataTree, DataArray,
            scalar, or array), they are simply assigned.
        **items_kwargs
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

        Returns
        -------
        dt : DataTree
            A new DataTree with the new variables or children in addition to all the
            existing items.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well-defined.
        Assigning multiple items within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        xarray.Dataset.assign
        pandas.DataFrame.assign
        """
        items = either_dict_or_kwargs(items, items_kwargs, 'assign')
        dt = self.copy()
        dt.update(items)
        return dt

    def drop_nodes(self: DataTree, names: str | Iterable[str], *, errors: ErrorOptions='raise') -> DataTree:
        """
        Drop child nodes from this node.

        Parameters
        ----------
        names : str or iterable of str
            Name(s) of nodes to drop.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a KeyError if any of the node names
            passed are not present as children of this node. If 'ignore',
            any given names that are present are dropped and no error is raised.

        Returns
        -------
        dropped : DataTree
            A copy of the node with the specified children dropped.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = {names}
        else:
            names = set(names)
        if errors == 'raise':
            extra = names - set(self.children)
            if extra:
                raise KeyError(f'Cannot drop all nodes - nodes {extra} not present')
        children_to_keep = {name: child for name, child in self.children.items() if name not in names}
        return self._replace(children=children_to_keep)

    @classmethod
    def from_dict(cls, d: MutableMapping[str, Dataset | DataArray | DataTree | None], name: str | None=None) -> DataTree:
        """
        Create a datatree from a dictionary of data objects, organised by paths into the tree.

        Parameters
        ----------
        d : dict-like
            A mapping from path names to xarray.Dataset, xarray.DataArray, or DataTree objects.

            Path names are to be given as unix-like path. If path names containing more than one part are given, new
            tree nodes will be constructed as necessary.

            To assign data to the root node of the tree use "/" as the path.
        name : Hashable | None, optional
            Name for the root node of the tree. Default is None.

        Returns
        -------
        DataTree

        Notes
        -----
        If your dictionary is nested you will need to flatten it before using this method.
        """
        root_data = d.pop('/', None)
        if isinstance(root_data, DataTree):
            obj = root_data.copy()
            obj.orphan()
        else:
            obj = cls(name=name, data=root_data, parent=None, children=None)
        if d:
            for path, data in d.items():
                node_name = NodePath(path).name
                if isinstance(data, DataTree):
                    new_node = data.copy()
                    new_node.orphan()
                else:
                    new_node = cls(name=node_name, data=data)
                obj._set_item(path, new_node, allow_overwrite=False, new_nodes_along_path=True)
        return obj

    def to_dict(self) -> dict[str, Dataset]:
        """
        Create a dictionary mapping of absolute node paths to the data contained in those nodes.

        Returns
        -------
        dict[str, Dataset]
        """
        return {node.path: node.to_dataset() for node in self.subtree}

    @property
    def nbytes(self) -> int:
        return sum((node.to_dataset().nbytes for node in self.subtree))

    def __len__(self) -> int:
        return len(self.children) + len(self.data_vars)

    @property
    def indexes(self) -> Indexes[pd.Index]:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this DataTree node has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        DataTree.xindexes
        """
        return self.xindexes.to_pandas_indexes()

    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self.to_dataset())

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        return DataVariables(self.to_dataset())

    def isomorphic(self, other: DataTree, from_root: bool=False, strict_names: bool=False) -> bool:
        """
        Two DataTrees are considered isomorphic if every node has the same number of children.

        Nothing about the data in each node is checked.

        Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
        such as ``tree1 + tree2``.

        By default this method does not check any part of the tree above the given node.
        Therefore this method can be used as default to check that two subtrees are isomorphic.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is False
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.
        strict_names : bool, optional, default is False
            Whether or not to also check that every node in the tree has the same name as its counterpart in the other
            tree.

        See Also
        --------
        DataTree.equals
        DataTree.identical
        """
        try:
            check_isomorphic(self, other, require_names_equal=strict_names, check_from_root=from_root)
            return True
        except (TypeError, TreeIsomorphismError):
            return False

    def equals(self, other: DataTree, from_root: bool=True) -> bool:
        """
        Two DataTrees are equal if they have isomorphic node structures, with matching node names,
        and if they have matching variables and coordinates, all of which are equal.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.equals
        DataTree.isomorphic
        DataTree.identical
        """
        if not self.isomorphic(other, from_root=from_root, strict_names=True):
            return False
        return all([node.ds.equals(other_node.ds) for node, other_node in zip(self.subtree, other.subtree)])

    def identical(self, other: DataTree, from_root=True) -> bool:
        """
        Like equals, but will also check all dataset attributes and the attributes on
        all variables and coordinates.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.identical
        DataTree.isomorphic
        DataTree.equals
        """
        if not self.isomorphic(other, from_root=from_root, strict_names=True):
            return False
        return all((node.ds.identical(other_node.ds) for node, other_node in zip(self.subtree, other.subtree)))

    def filter(self: DataTree, filterfunc: Callable[[DataTree], bool]) -> DataTree:
        """
        Filter nodes according to a specified condition.

        Returns a new tree containing only the nodes in the original tree for which `fitlerfunc(node)` is True.
        Will also contain empty nodes at intermediate positions if required to support leaves.

        Parameters
        ----------
        filterfunc: function
            A function which accepts only one DataTree - the node on which filterfunc will be called.

        Returns
        -------
        DataTree

        See Also
        --------
        match
        pipe
        map_over_subtree
        """
        filtered_nodes = {node.path: node.ds for node in self.subtree if filterfunc(node)}
        return DataTree.from_dict(filtered_nodes, name=self.root.name)

    def match(self, pattern: str) -> DataTree:
        """
        Return nodes with paths matching pattern.

        Uses unix glob-like syntax for pattern-matching.

        Parameters
        ----------
        pattern: str
            A pattern to match each node path against.

        Returns
        -------
        DataTree

        See Also
        --------
        filter
        pipe
        map_over_subtree

        Examples
        --------
        >>> dt = DataTree.from_dict(
        ...     {
        ...         "/a/A": None,
        ...         "/a/B": None,
        ...         "/b/A": None,
        ...         "/b/B": None,
        ...     }
        ... )
        >>> dt.match("*/B")
        DataTree('None', parent=None)
        ├── DataTree('a')
        │   └── DataTree('B')
        └── DataTree('b')
            └── DataTree('B')
        """
        matching_nodes = {node.path: node.ds for node in self.subtree if NodePath(node.path).match(pattern)}
        return DataTree.from_dict(matching_nodes, name=self.root.name)

    def map_over_subtree(self, func: Callable, *args: Iterable[Any], **kwargs: Any) -> DataTree | tuple[DataTree]:
        """
        Apply a function to every dataset in this subtree, returning a new tree which stores the results.

        The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
        descendant nodes. The returned tree will have the same structure as the original subtree.

        func needs to return a Dataset in order to rebuild the subtree.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        subtrees : DataTree, tuple of DataTrees
            One or more subtrees containing results from applying ``func`` to the data at each node.
        """
        return map_over_subtree(func)(self, *args, **kwargs)

    def map_over_subtree_inplace(self, func: Callable, *args: Iterable[Any], **kwargs: Any) -> None:
        """
        Apply a function to every dataset in this subtree, updating data in place.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets,
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.
        """
        for node in self.subtree:
            if node.has_data:
                node.ds = func(node.ds, *args, **kwargs)

    def pipe(self, func: Callable | tuple[Callable, str], *args: Any, **kwargs: Any) -> Any:
        """Apply ``func(self, *args, **kwargs)``

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : callable
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        *args
            positional arguments passed into ``func``.
        **kwargs
            a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : Any
            the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        .. code:: python

            f(g(h(dt), arg1=a), arg2=b, arg3=c)

        You can write

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe((f, "arg2"), arg1=a, arg3=c))

        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError(f'{target} is both the pipe target and a keyword argument')
            kwargs[target] = self
        else:
            args = (self,) + args
        return func(*args, **kwargs)

    def render(self):
        """Print tree structure, including any data stored at each node."""
        for pre, fill, node in RenderTree(self):
            print(f"{pre}DataTree('{self.name}')")
            for ds_line in repr(node.ds)[1:]:
                print(f'{fill}{ds_line}')

    def merge(self, datatree: DataTree) -> DataTree:
        """Merge all the leaves of a second DataTree into this one."""
        raise NotImplementedError

    def merge_child_nodes(self, *paths, new_path: T_Path) -> DataTree:
        """Merge a set of child nodes into a single new node."""
        raise NotImplementedError

    def to_dataarray(self) -> DataArray:
        return self.ds.to_dataarray()

    @property
    def groups(self):
        """Return all netCDF4 groups in the tree, given as a tuple of path-like strings."""
        return tuple((node.path for node in self.subtree))

    def to_netcdf(self, filepath, mode: str='w', encoding=None, unlimited_dims=None, **kwargs):
        """
        Write datatree contents to a netCDF file.

        Parameters
        ----------
        filepath : str or Path
            Path to which to save this datatree.
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten. Only appies to the root group.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}, ...}``. See ``xarray.Dataset.to_netcdf`` for available
            options.
        unlimited_dims : dict, optional
            Mapping of unlimited dimensions per group that that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding["unlimited_dims"]``.
        kwargs :
            Addional keyword arguments to be passed to ``xarray.Dataset.to_netcdf``
        """
        from xarray.datatree_.datatree.io import _datatree_to_netcdf
        _datatree_to_netcdf(self, filepath, mode=mode, encoding=encoding, unlimited_dims=unlimited_dims, **kwargs)

    def to_zarr(self, store, mode: str='w-', encoding=None, consolidated: bool=True, **kwargs):
        """
        Write datatree contents to a Zarr store.

        Parameters
        ----------
        store : MutableMapping, str or Path, optional
            Store or path to directory in file system
        mode : {{"w", "w-", "a", "r+", None}, default: "w-"
            Persistence mode: “w” means create (overwrite if exists); “w-” means create (fail if exists);
            “a” means override existing variables (create if does not exist); “r+” means modify existing
            array values only (raise an error if any metadata or shapes would change). The default mode
            is “w-”.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        consolidated : bool
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing metadata for all groups.
        kwargs :
            Additional keyword arguments to be passed to ``xarray.Dataset.to_zarr``
        """
        from xarray.datatree_.datatree.io import _datatree_to_zarr
        _datatree_to_zarr(self, store, mode=mode, encoding=encoding, consolidated=consolidated, **kwargs)

    def plot(self):
        raise NotImplementedError