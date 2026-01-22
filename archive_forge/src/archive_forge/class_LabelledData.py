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
class LabelledData(param.Parameterized):
    """
    LabelledData is a mix-in class designed to introduce the group and
    label parameters (and corresponding methods) to any class
    containing data. This class assumes that the core data contents
    will be held in the attribute called 'data'.

    Used together, group and label are designed to allow a simple and
    flexible means of addressing data. For instance, if you are
    collecting the heights of people in different demographics, you
    could specify the values of your objects as 'Height' and then use
    the label to specify the (sub)population.

    In this scheme, one object may have the parameters set to
    [group='Height', label='Children'] and another may use
    [group='Height', label='Adults'].

    Note: Another level of specification is implicit in the type (i.e
    class) of the LabelledData object. A full specification of a
    LabelledData object is therefore given by the tuple
    (<type>, <group>, label>). This additional level of specification is
    used in the traverse method.

    Any strings can be used for the group and label, but it can be
    convenient to use a capitalized string of alphanumeric characters,
    in which case the keys used for matching in the matches and
    traverse method will correspond exactly to {type}.{group}.{label}.
    Otherwise the strings provided will be sanitized to be valid
    capitalized Python identifiers, which works fine but can sometimes
    be confusing.
    """
    group = param.String(default='LabelledData', constant=True, doc='\n       A string describing the type of data contained by the object.\n       By default this will typically mirror the class name.')
    label = param.String(default='', constant=True, doc='\n       Optional label describing the data, typically reflecting where\n       or how it was measured. The label should allow a specific\n       measurement or dataset to be referenced for a given group.')
    _deep_indexable = False

    def __init__(self, data, id=None, plot_id=None, **params):
        """
        All LabelledData subclasses must supply data to the
        constructor, which will be held on the .data attribute.
        This class also has an id instance attribute, which
        may be set to associate some custom options with the object.
        """
        self.data = data
        self._id = None
        self.id = id
        self._plot_id = plot_id or builtins.id(self)
        if isinstance(params.get('label', None), tuple):
            alias, long_name = params['label']
            util.label_sanitizer.add_aliases(**{alias: long_name})
            params['label'] = long_name
        if isinstance(params.get('group', None), tuple):
            alias, long_name = params['group']
            util.group_sanitizer.add_aliases(**{alias: long_name})
            params['group'] = long_name
        super().__init__(**params)
        if not util.group_sanitizer.allowable(self.group):
            raise ValueError('Supplied group %r contains invalid characters.' % self.group)
        elif not util.label_sanitizer.allowable(self.label):
            raise ValueError('Supplied label %r contains invalid characters.' % self.label)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, opts_id):
        """Handles tracking and cleanup of custom ids."""
        old_id = self._id
        self._id = opts_id
        if old_id is not None:
            cleanup_custom_options(old_id)
        if opts_id is not None and opts_id != old_id:
            if opts_id not in Store._weakrefs:
                Store._weakrefs[opts_id] = []
            ref = weakref.ref(self, partial(cleanup_custom_options, opts_id))
            Store._weakrefs[opts_id].append(ref)

    def clone(self, data=None, shared_data=True, new_type=None, link=True, *args, **overrides):
        """Clones the object, overriding data and parameters.

        Args:
            data: New data replacing the existing data
            shared_data (bool, optional): Whether to use existing data
            new_type (optional): Type to cast object to
            link (bool, optional): Whether clone should be linked
                Determines whether Streams and Links attached to
                original object will be inherited.
            *args: Additional arguments to pass to constructor
            **overrides: New keyword arguments to pass to constructor

        Returns:
            Cloned object
        """
        params = self.param.values()
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.param.objects('existing')
            params = {k: v for k, v in params.items() if k in new_params}
            if params.get('group') == self.param.objects('existing')['group'].default:
                params.pop('group')
        settings = dict(params, **overrides)
        if 'id' not in settings:
            settings['id'] = self.id
        if data is None and shared_data:
            data = self.data
            if link:
                settings['plot_id'] = self._plot_id
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        return clone_type(data, *args, **{k: v for k, v in settings.items() if k not in pos_args})

    def relabel(self, label=None, group=None, depth=0):
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
        new_data = self.data
        if depth > 0 and getattr(self, '_deep_indexable', False):
            new_data = []
            for k, v in self.data.items():
                relabelled = v.relabel(group=group, label=label, depth=depth - 1)
                new_data.append((k, relabelled))
        keywords = [('label', label), ('group', group)]
        kwargs = {k: v for k, v in keywords if v is not None}
        return self.clone(new_data, **kwargs)

    def matches(self, spec):
        """Whether the spec applies to this object.

        Args:
            spec: A function, spec or type to check for a match
                * A 'type[[.group].label]' string which is compared
                  against the type, group and label of this object
                * A function which is given the object and returns
                  a boolean.
                * An object type matched using isinstance.

        Returns:
            bool: Whether the spec matched this object.
        """
        if callable(spec) and (not isinstance(spec, type)):
            return spec(self)
        elif isinstance(spec, type):
            return isinstance(self, spec)
        specification = (self.__class__.__name__, self.group, self.label)
        split_spec = tuple(spec.split('.')) if not isinstance(spec, tuple) else spec
        split_spec, nocompare = zip(*((None, True) if s == '*' or s is None else (s, False) for s in split_spec))
        if all(nocompare):
            return True
        match_fn = itemgetter(*(idx for idx, nc in enumerate(nocompare) if not nc))
        self_spec = match_fn(split_spec)
        unescaped_match = match_fn(specification[:len(split_spec)]) == self_spec
        if unescaped_match:
            return True
        sanitizers = [util.sanitize_identifier, util.group_sanitizer, util.label_sanitizer]
        identifier_specification = tuple((fn(ident, escape=False) for ident, fn in zip(specification, sanitizers)))
        identifier_match = match_fn(identifier_specification[:len(split_spec)]) == self_spec
        return identifier_match

    def traverse(self, fn=None, specs=None, full_breadth=True):
        """Traverses object returning matching items
        Traverses the set of children of the object, collecting the
        all objects matching the defined specs. Each object can be
        processed with the supplied function.
        Args:
            fn (function, optional): Function applied to matched objects
            specs: List of specs to match
                Specs must be types, functions or type[.group][.label]
                specs to select objects to return, by default applies
                to all objects.
            full_breadth: Whether to traverse all objects
                Whether to traverse the full set of objects on each
                container or only the first.
        Returns:
            list: List of objects that matched
        """
        if fn is None:
            fn = lambda x: x
        if specs is not None and (not isinstance(specs, (list, set, tuple))):
            specs = [specs]
        accumulator = []
        matches = specs is None
        if not matches:
            for spec in specs:
                matches = self.matches(spec)
                if matches:
                    break
        if matches:
            accumulator.append(fn(self))
        if self._deep_indexable:
            for el in self:
                if el is None:
                    continue
                accumulator += el.traverse(fn, specs, full_breadth)
                if not full_breadth:
                    break
        return accumulator

    def map(self, map_fn, specs=None, clone=True):
        """Map a function to all objects matching the specs

        Recursively replaces elements using a map function when the
        specs apply, by default applies to all objects, e.g. to apply
        the function to all contained Curve objects:

            dmap.map(fn, hv.Curve)

        Args:
            map_fn: Function to apply to each object
            specs: List of specs to match
                List of types, functions or type[.group][.label] specs
                to select objects to return, by default applies to all
                objects.
            clone: Whether to clone the object or transform inplace

        Returns:
            Returns the object after the map_fn has been applied
        """
        if specs is not None and (not isinstance(specs, (list, set, tuple))):
            specs = [specs]
        applies = specs is None or any((self.matches(spec) for spec in specs))
        if self._deep_indexable:
            deep_mapped = self.clone(shared_data=False) if clone else self
            for k, v in self.items():
                new_val = v.map(map_fn, specs, clone)
                if new_val is not None:
                    deep_mapped[k] = new_val
            if applies:
                deep_mapped = map_fn(deep_mapped)
            return deep_mapped
        else:
            return map_fn(self) if applies else self

    def __getstate__(self):
        """Ensures pickles save options applied to this objects."""
        obj_dict = self.__dict__.copy()
        try:
            if Store.save_option_state and obj_dict.get('_id', None) is not None:
                custom_key = '_custom_option_%d' % obj_dict['_id']
                if custom_key not in obj_dict:
                    obj_dict[custom_key] = {backend: s[obj_dict['_id']] for backend, s in Store._custom_options.items() if obj_dict['_id'] in s}
            else:
                obj_dict['_id'] = None
        except Exception:
            self.param.warning('Could not pickle custom style information.')
        return obj_dict

    def __setstate__(self, d):
        """Restores options applied to this object."""
        d = param_aliases(d)
        opts_id = d['_id'] if '_id' in d else d.pop('id', None)
        try:
            load_options = Store.load_counter_offset is not None
            if load_options:
                matches = [k for k in d if k.startswith('_custom_option')]
                for match in matches:
                    custom_id = int(match.split('_')[-1]) + Store.load_counter_offset
                    if not isinstance(d[match], dict):
                        backend_info = {'matplotlib': d[match]}
                    else:
                        backend_info = d[match]
                    for backend, info in backend_info.items():
                        if backend not in Store._custom_options:
                            Store._custom_options[backend] = {}
                        Store._custom_options[backend][custom_id] = info
                    if backend_info:
                        if custom_id not in Store._weakrefs:
                            Store._weakrefs[custom_id] = []
                        ref = weakref.ref(self, partial(cleanup_custom_options, custom_id))
                        Store._weakrefs[opts_id].append(ref)
                    d.pop(match)
                if opts_id is not None:
                    opts_id += Store.load_counter_offset
        except Exception:
            self.param.warning('Could not unpickle custom style information.')
        d['_id'] = opts_id
        self.__dict__.update(d)
        super().__setstate__({})