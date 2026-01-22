import inspect
import os
import shutil
import sys
from collections import defaultdict
from inspect import Parameter, Signature
from pathlib import Path
from types import FunctionType
import param
from pyviz_comms import extension as _pyviz_extension
from ..core import (
from ..core.operation import Operation, OperationCallable
from ..core.options import Keywords, Options, options_policy
from ..core.overlay import Overlay
from ..core.util import merge_options_to_dict
from ..operation.element import function
from ..streams import Params, Stream, streams_list_from_dict
from .settings import OutputSettings, list_backends, list_formats
class opts(param.ParameterizedFunction, metaclass=OptsMeta):
    """
    Utility function to set options at the global level or to provide an
    Options object that can be used with the .options method of an
    element or container.

    Option objects can be generated and validated in a tab-completable
    way (in appropriate environments such as Jupyter notebooks) using
    completers such as opts.Curve, opts.Image, opts.Overlay, etc.

    To set opts globally you can pass these option objects into opts.defaults:

    opts.defaults(*options)

    For instance:

    opts.defaults(opts.Curve(color='red'))

    To set opts on a specific object, you can supply these option
    objects to the .options method.

    For instance:

    curve = hv.Curve([1,2,3])
    curve.options(opts.Curve(color='red'))

    The options method also accepts lists of Option objects.
    """
    __original_docstring__ = None
    _no_completion = ['title_format', 'color_index', 'size_index', 'scaling_factor', 'scaling_method', 'size_fn', 'normalize_lengths', 'group_index', 'category_index', 'stack_index', 'color_by']
    strict = param.Boolean(default=False, doc='\n       Whether to be strict about the options specification. If not set\n       to strict (default), any invalid keywords are simply skipped. If\n       strict, invalid keywords prevent the options being applied.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **params):
        if not params and (not args):
            return Options()
        elif params and (not args):
            return Options(**params)

    @classmethod
    def _group_kwargs_to_options(cls, obj, kwargs):
        """Format option group kwargs into canonical options format"""
        groups = Options._option_groups
        if set(kwargs.keys()) - set(groups):
            raise Exception('Keyword options {} must be one of  {}'.format(groups, ','.join((repr(g) for g in groups))))
        elif not all((isinstance(v, dict) for v in kwargs.values())):
            raise Exception('The %s options must be specified using dictionary groups' % ','.join((repr(k) for k in kwargs.keys())))
        targets = [grp and all((k[0].isupper() for k in grp)) for grp in kwargs.values()]
        if any(targets) and (not all(targets)):
            raise Exception("Cannot mix target specification keys such as 'Image' with non-target keywords.")
        elif not any(targets):
            sanitized_group = util.group_sanitizer(obj.group)
            if obj.label:
                identifier = '{}.{}.{}'.format(obj.__class__.__name__, sanitized_group, util.label_sanitizer(obj.label))
            elif sanitized_group != obj.__class__.__name__:
                identifier = f'{obj.__class__.__name__}.{sanitized_group}'
            else:
                identifier = obj.__class__.__name__
            options = {identifier: {grp: kws for grp, kws in kwargs.items()}}
        else:
            dfltdict = defaultdict(dict)
            for grp, entries in kwargs.items():
                for identifier, kws in entries.items():
                    dfltdict[identifier][grp] = kws
            options = dict(dfltdict)
        return options

    @classmethod
    def _apply_groups_to_backend(cls, obj, options, backend, clone):
        """Apply the groups to a single specified backend"""
        obj_handle = obj
        if options is None:
            if clone:
                obj_handle = obj.map(lambda x: x.clone(id=None))
            else:
                obj.map(lambda x: setattr(x, 'id', None))
        elif clone:
            obj_handle = obj.map(lambda x: x.clone(id=x.id))
        return StoreOptions.set_options(obj_handle, options, backend=backend)

    @classmethod
    def _grouped_backends(cls, options, backend):
        """Group options by backend and filter out output group appropriately"""
        if options is None:
            return [(backend or Store.current_backend, options)]
        dfltdict = defaultdict(dict)
        for spec, groups in options.items():
            if 'output' not in groups.keys() or len(groups['output']) == 0:
                dfltdict[backend or Store.current_backend][spec.strip()] = groups
            elif set(groups['output'].keys()) - {'backend'}:
                dfltdict[groups['output']['backend']][spec.strip()] = groups
            elif ['backend'] == list(groups['output'].keys()):
                filtered = {k: v for k, v in groups.items() if k != 'output'}
                dfltdict[groups['output']['backend']][spec.strip()] = filtered
            else:
                raise Exception('The output options group must have the backend keyword')
        return [(bk, bk_opts) for bk, bk_opts in dfltdict.items()]

    @classmethod
    def apply_groups(cls, obj, options=None, backend=None, clone=True, **kwargs):
        """Applies nested options definition grouped by type.

        Applies options on an object or nested group of objects,
        returning a new object with the options applied. This method
        accepts the separate option namespaces explicitly (i.e. 'plot',
        'style', and 'norm').

        If the options are to be set directly on the object a
        simple format may be used, e.g.:

            opts.apply_groups(obj, style={'cmap': 'viridis'},
                                         plot={'show_title': False})

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            opts.apply_groups(obj, {'Image': {'plot':  {'show_title': False},
                                              'style': {'cmap': 'viridis}}})

        If no opts are supplied all options on the object will be reset.

        Args:
            options (dict): Options specification
                Options specification should be indexed by
                type[.group][.label] or option type ('plot', 'style',
                'norm').
            backend (optional): Backend to apply options to
                Defaults to current selected backend
            clone (bool, optional): Whether to clone object
                Options can be applied inplace with clone=False
            **kwargs: Keywords of options by type
                Applies options directly to the object by type
                (e.g. 'plot', 'style', 'norm') specified as
                dictionaries.

        Returns:
            Returns the object or a clone with the options applied
        """
        if isinstance(options, str):
            from ..util.parser import OptsSpec
            try:
                options = OptsSpec.parse(options)
            except SyntaxError:
                options = OptsSpec.parse(f'{obj.__class__.__name__} {options}')
        if kwargs:
            options = cls._group_kwargs_to_options(obj, kwargs)
        for backend_loop, backend_opts in cls._grouped_backends(options, backend):
            obj = cls._apply_groups_to_backend(obj, backend_opts, backend_loop, clone)
        return obj

    @classmethod
    def _process_magic(cls, options, strict, backends=None):
        if isinstance(options, str):
            from .parser import OptsSpec
            try:
                ns = get_ipython().user_ns
            except Exception:
                ns = globals()
            options = OptsSpec.parse(options, ns=ns)
        errmsg = StoreOptions.validation_error_message(options, backends=backends)
        if errmsg:
            sys.stderr.write(errmsg)
            if strict:
                sys.stderr.write('Options specification will not be applied.')
                return (options, True)
        return (options, False)

    @classmethod
    def _linemagic(cls, options, strict=False, backend=None):
        backends = None if backend is None else [backend]
        options, failure = cls._process_magic(options, strict, backends=backends)
        if failure:
            return
        with options_policy(skip_invalid=True, warn_on_skip=False):
            StoreOptions.apply_customizations(options, Store.options(backend=backend))

    @classmethod
    def defaults(cls, *options, **kwargs):
        """Set default options for a session.

        Set default options for a session. whether in a Python script or
        a Jupyter notebook.

        Args:
           *options: Option objects used to specify the defaults.
           backend:  The plotting extension the options apply to
        """
        if kwargs and len(kwargs) != 1 and (next(iter(kwargs.keys())) != 'backend'):
            raise Exception('opts.defaults only accepts "backend" keyword argument')
        cls._linemagic(cls._expand_options(merge_options_to_dict(options)), backend=kwargs.get('backend'))

    @classmethod
    def _expand_by_backend(cls, options, backend):
        """
        Given a list of flat Option objects which may or may not have
        'backend' in their kwargs, return a list of grouped backend
        """
        groups = defaultdict(list)
        used_fallback = False
        for obj in options:
            if 'backend' in obj.kwargs:
                opts_backend = obj.kwargs['backend']
            elif backend is None:
                opts_backend = Store.current_backend
                obj.kwargs['backend'] = opts_backend
            else:
                opts_backend = backend
                obj.kwargs['backend'] = opts_backend
                used_fallback = True
            groups[opts_backend].append(obj)
        if backend and (not used_fallback):
            cls.param.warning('All supplied Options objects already define a backend, backend override %r will be ignored.' % backend)
        return [(bk, cls._expand_options(o, bk)) for bk, o in groups.items()]

    @classmethod
    def _expand_options(cls, options, backend=None):
        """
        Validates and expands a dictionaries of options indexed by
        type[.group][.label] keys into separate style, plot, norm and
        output options.

            opts._expand_options({'Image': dict(cmap='viridis', show_title=False)})

        returns

            {'Image': {'plot': dict(show_title=False), 'style': dict(cmap='viridis')}}
        """
        current_backend = Store.current_backend
        if not Store.renderers:
            raise ValueError('No plotting extension is currently loaded. Ensure you load an plotting extension with hv.extension or import it explicitly from holoviews.plotting before applying any options.')
        elif current_backend not in Store.renderers:
            raise ValueError('Currently selected plotting extension {ext} has not been loaded, ensure you load it with hv.extension({ext}) before setting options'.format(ext=repr(current_backend)))
        try:
            backend_options = Store.options(backend=backend or current_backend)
        except KeyError as e:
            raise Exception(f'The {e} backend is not loaded. Please load the backend using hv.extension.') from None
        expanded = {}
        if isinstance(options, list):
            options = merge_options_to_dict(options)
        for objspec, option_values in options.items():
            objtype = objspec.split('.')[0]
            if objtype not in backend_options:
                raise ValueError(f'{objtype} type not found, could not apply options.')
            obj_options = backend_options[objtype]
            expanded[objspec] = {g: {} for g in obj_options.groups}
            for opt, value in option_values.items():
                for g, group_opts in sorted(obj_options.groups.items()):
                    if opt in group_opts.allowed_keywords:
                        expanded[objspec][g][opt] = value
                        break
                else:
                    valid_options = sorted({keyword for group_opts in obj_options.groups.values() for keyword in group_opts.allowed_keywords})
                    cls._options_error(opt, objtype, backend, valid_options)
        return expanded

    @classmethod
    def _options_error(cls, opt, objtype, backend, valid_options):
        """
        Generates an error message for an invalid option suggesting
        similar options through fuzzy matching.
        """
        current_backend = Store.current_backend
        loaded_backends = Store.loaded_backends()
        kws = Keywords(values=valid_options)
        matches = sorted(kws.fuzzy_match(opt))
        if backend is not None:
            if matches:
                raise ValueError(f'Unexpected option {opt!r} for {objtype} type when using the {backend!r} extension. Similar options are: {matches}.')
            else:
                raise ValueError(f'Unexpected option {opt!r} for {objtype} type when using the {backend!r} extension. No similar options found.')
        found = []
        for lb in [b for b in loaded_backends if b != backend]:
            lb_options = Store.options(backend=lb).get(objtype)
            if lb_options is None:
                continue
            for _g, group_opts in lb_options.groups.items():
                if opt in group_opts.allowed_keywords:
                    found.append(lb)
        if found:
            param.main.param.warning(f'Option {opt!r} for {objtype} type not valid for selected backend ({current_backend!r}). Option only applies to following backends: {found!r}')
            return
        if matches:
            raise ValueError(f'Unexpected option {opt!r} for {objtype} type across all extensions. Similar options for current extension ({current_backend!r}) are: {matches}.')
        else:
            raise ValueError(f'Unexpected option {opt!r} for {objtype} type across all extensions. No similar options found.')

    @classmethod
    def _builder_reprs(cls, options, namespace=None, ns=None):
        """
        Given a list of Option objects (such as those returned from
        OptsSpec.parse_options) or an %opts or %%opts magic string,
        return a list of corresponding option builder reprs. The
        namespace is typically given as 'hv' if fully qualified
        namespaces are desired.
        """
        if isinstance(options, str):
            from .parser import OptsSpec
            if ns is None:
                try:
                    ns = get_ipython().user_ns
                except Exception:
                    ns = globals()
            options = options.replace('%%opts', '').replace('%opts', '')
            options = OptsSpec.parse_options(options, ns=ns)
        reprs = []
        ns = f'{namespace}.' if namespace else ''
        for option in options:
            kws = ', '.join((f'{k}={option.kwargs[k]!r}' for k in sorted(option.kwargs)))
            if '.' in option.key:
                element = option.key.split('.')[0]
                spec = repr('.'.join(option.key.split('.')[1:])) + ', '
            else:
                element = option.key
                spec = ''
            opts_format = '{ns}opts.{element}({spec}{kws})'
            reprs.append(opts_format.format(ns=ns, spec=spec, kws=kws, element=element))
        return reprs

    @classmethod
    def _create_builder(cls, element, completions):

        def builder(cls, spec=None, **kws):
            spec = element if spec is None else f'{element}.{spec}'
            prefix = f'In opts.{element}(...), '
            backend = kws.get('backend', None)
            keys = set(kws.keys())
            if backend:
                allowed_kws = cls._element_keywords(backend, elements=[element])[element]
                invalid = keys - set(allowed_kws)
            else:
                mismatched = {}
                all_valid_kws = set()
                for loaded_backend in Store.loaded_backends():
                    valid = set(cls._element_keywords(loaded_backend).get(element, []))
                    all_valid_kws |= set(valid)
                    if keys <= valid:
                        return Options(spec, **kws)
                    mismatched[loaded_backend] = list(keys - valid)
                invalid = keys - all_valid_kws
                if mismatched and (not invalid):
                    msg = '{prefix}keywords supplied are mixed across backends. Keyword(s) {info}'
                    info = ', '.join(('{} are invalid for {}'.format(', '.join((repr(el) for el in v)), k) for k, v in mismatched.items()))
                    raise ValueError(msg.format(info=info, prefix=prefix))
                allowed_kws = completions
            reraise = False
            if invalid:
                try:
                    cls._options_error(next(iter(invalid)), element, backend, allowed_kws)
                except ValueError as e:
                    msg = str(e)[0].lower() + str(e)[1:]
                    reraise = True
                if reraise:
                    raise ValueError(prefix + msg)
            return Options(spec, **kws)
        filtered_keywords = [k for k in completions if k not in cls._no_completion]
        sorted_kw_set = sorted(set(filtered_keywords))
        signature = Signature([Parameter('spec', Parameter.POSITIONAL_OR_KEYWORD)] + [Parameter(kw, Parameter.KEYWORD_ONLY) for kw in sorted_kw_set])
        builder.__signature__ = signature
        return classmethod(builder)

    @classmethod
    def _element_keywords(cls, backend, elements=None):
        """Returns a dictionary of element names to allowed keywords"""
        if backend not in Store.loaded_backends():
            return {}
        mapping = {}
        backend_options = Store.options(backend)
        elements = elements if elements is not None else backend_options.keys()
        for element in elements:
            if '.' in element:
                continue
            element = element if isinstance(element, tuple) else (element,)
            element_keywords = []
            options = backend_options['.'.join(element)]
            for group in Options._option_groups:
                element_keywords.extend(options[group].allowed_keywords)
            mapping[element[0]] = element_keywords
        return mapping

    @classmethod
    def _update_backend(cls, backend):
        if cls.__original_docstring__ is None:
            cls.__original_docstring__ = cls.__doc__
        all_keywords = set()
        element_keywords = cls._element_keywords(backend)
        for element, keywords in element_keywords.items():
            with param.logging_level('CRITICAL'):
                all_keywords |= set(keywords)
                setattr(cls, element, cls._create_builder(element, keywords))
        filtered_keywords = [k for k in all_keywords if k not in cls._no_completion]
        sorted_kw_set = sorted(set(filtered_keywords))
        from inspect import Parameter, Signature
        signature = Signature([Parameter('args', Parameter.VAR_POSITIONAL)] + [Parameter(kw, Parameter.KEYWORD_ONLY) for kw in sorted_kw_set])
        cls.__init__.__signature__ = signature