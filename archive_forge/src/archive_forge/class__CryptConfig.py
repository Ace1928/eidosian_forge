from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
class _CryptConfig(object):
    """parses, validates, and stores CryptContext config

    this is a helper used internally by CryptContext to handle
    parsing, validation, and serialization of its config options.
    split out from the main class, but not made public since
    that just complicates interface too much (c.f. CryptPolicy)

    :arg source: config as dict mapping ``(cat,scheme,option) -> value``
    """
    _scheme_options = None
    _context_options = None
    handlers = None
    schemes = None
    categories = None
    context_kwds = None
    _default_schemes = None
    _records = None
    _record_lists = None

    def __init__(self, source):
        self._init_scheme_list(source.get((None, None, 'schemes')))
        self._init_options(source)
        self._init_default_schemes()
        self._init_records()

    def _init_scheme_list(self, data):
        """initialize .handlers and .schemes attributes"""
        handlers = []
        schemes = []
        if isinstance(data, native_string_types):
            data = splitcomma(data)
        for elem in data or ():
            if hasattr(elem, 'name'):
                handler = elem
                scheme = handler.name
                _validate_handler_name(scheme)
            elif isinstance(elem, native_string_types):
                handler = get_crypt_handler(elem)
                scheme = handler.name
            else:
                raise TypeError('scheme must be name or CryptHandler, not %r' % type(elem))
            if scheme in schemes:
                raise KeyError('multiple handlers with same name: %r' % (scheme,))
            handlers.append(handler)
            schemes.append(scheme)
        self.handlers = tuple(handlers)
        self.schemes = tuple(schemes)

    def _init_options(self, source):
        """load config dict into internal representation,
        and init .categories attr
        """
        norm_scheme_option = self._norm_scheme_option
        norm_context_option = self._norm_context_option
        self._scheme_options = scheme_options = {}
        self._context_options = context_options = {}
        categories = set()
        for (cat, scheme, key), value in iteritems(source):
            categories.add(cat)
            explicit_scheme = scheme
            if not cat and (not scheme) and (key in _global_settings):
                scheme = 'all'
            if scheme:
                key, value = norm_scheme_option(key, value)
                if scheme == 'all' and key not in _global_settings:
                    warn("The '%s' option should be configured per-algorithm, and not set globally in the context; This will be an error in Passlib 2.0" % (key,), PasslibConfigWarning)
                if explicit_scheme == 'all':
                    warn("The 'all' scheme is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0; Please configure options on a per-algorithm basis.", DeprecationWarning)
                try:
                    category_map = scheme_options[scheme]
                except KeyError:
                    scheme_options[scheme] = {cat: {key: value}}
                else:
                    try:
                        option_map = category_map[cat]
                    except KeyError:
                        category_map[cat] = {key: value}
                    else:
                        option_map[key] = value
            else:
                if cat and key == 'schemes':
                    raise KeyError("'schemes' context option is not allowed per category")
                key, value = norm_context_option(cat, key, value)
                if key == 'min_verify_time':
                    continue
                try:
                    category_map = context_options[key]
                except KeyError:
                    context_options[key] = {cat: value}
                else:
                    category_map[cat] = value
        categories.discard(None)
        self.categories = tuple(sorted(categories))

    def _norm_scheme_option(self, key, value):
        if key in _forbidden_scheme_options:
            raise KeyError('%r option not allowed in CryptContext configuration' % (key,))
        if isinstance(value, native_string_types):
            func = _coerce_scheme_options.get(key)
            if func:
                value = func(value)
        return (key, value)

    def _norm_context_option(self, cat, key, value):
        schemes = self.schemes
        if key == 'default':
            if hasattr(value, 'name'):
                value = value.name
            elif not isinstance(value, native_string_types):
                raise ExpectedTypeError(value, 'str', 'default')
            if schemes and value not in schemes:
                raise KeyError('default scheme not found in policy')
        elif key == 'deprecated':
            if isinstance(value, native_string_types):
                value = splitcomma(value)
            elif not isinstance(value, (list, tuple)):
                raise ExpectedTypeError(value, 'str or seq', 'deprecated')
            if 'auto' in value:
                if len(value) > 1:
                    raise ValueError("cannot list other schemes if ``deprecated=['auto']`` is used")
            elif schemes:
                for scheme in value:
                    if not isinstance(scheme, native_string_types):
                        raise ExpectedTypeError(value, 'str', 'deprecated element')
                    if scheme not in schemes:
                        raise KeyError('deprecated scheme not found in policy: %r' % (scheme,))
        elif key == 'min_verify_time':
            warn("'min_verify_time' was deprecated in Passlib 1.6, is ignored in 1.7, and will be removed in 1.8", DeprecationWarning)
        elif key == 'harden_verify':
            warn("'harden_verify' is deprecated & ignored as of Passlib 1.7.1,  and will be removed in 1.8", DeprecationWarning)
        elif key != 'schemes':
            raise KeyError('unknown CryptContext keyword: %r' % (key,))
        return (key, value)

    def get_context_optionmap(self, key, _default={}):
        """return dict mapping category->value for specific context option.

        .. warning:: treat return value as readonly!
        """
        return self._context_options.get(key, _default)

    def get_context_option_with_flag(self, category, key):
        """return value of specific option, handling category inheritance.
        also returns flag indicating whether value is category-specific.
        """
        try:
            category_map = self._context_options[key]
        except KeyError:
            return (None, False)
        value = category_map.get(None)
        if category:
            try:
                alt = category_map[category]
            except KeyError:
                pass
            else:
                if value is None or alt != value:
                    return (alt, True)
        return (value, False)

    def _get_scheme_optionmap(self, scheme, category, default={}):
        """return all options for (scheme,category) combination

        .. warning:: treat return value as readonly!
        """
        try:
            return self._scheme_options[scheme][category]
        except KeyError:
            return default

    def get_base_handler(self, scheme):
        return self.handlers[self.schemes.index(scheme)]

    @staticmethod
    def expand_settings(handler):
        setting_kwds = handler.setting_kwds
        if 'rounds' in handler.setting_kwds:
            setting_kwds += uh.HasRounds.using_rounds_kwds
        return setting_kwds

    def get_scheme_options_with_flag(self, scheme, category):
        """return composite dict of all options set for scheme.
        includes options inherited from 'all' and from default category.
        result can be modified.
        returns (kwds, has_cat_specific_options)
        """
        get_optionmap = self._get_scheme_optionmap
        kwds = get_optionmap('all', None).copy()
        has_cat_options = False
        if category:
            defkwds = kwds.copy()
            kwds.update(get_optionmap('all', category))
        allowed_settings = self.expand_settings(self.get_base_handler(scheme))
        for key in set(kwds).difference(allowed_settings):
            kwds.pop(key)
        if category:
            for key in set(defkwds).difference(allowed_settings):
                defkwds.pop(key)
        other = get_optionmap(scheme, None)
        kwds.update(other)
        if category:
            defkwds.update(other)
            kwds.update(get_optionmap(scheme, category))
            if kwds != defkwds:
                has_cat_options = True
        return (kwds, has_cat_options)

    def _init_default_schemes(self):
        """initialize maps containing default scheme for each category.

        have to do this after _init_options(), since the default scheme
        is affected by the list of deprecated schemes.
        """
        get_optionmap = self.get_context_optionmap
        default_map = self._default_schemes = get_optionmap('default').copy()
        dep_map = get_optionmap('deprecated')
        schemes = self.schemes
        if not schemes:
            return
        deps = dep_map.get(None) or ()
        default = default_map.get(None)
        if not default:
            for scheme in schemes:
                if scheme not in deps:
                    default_map[None] = scheme
                    break
            else:
                raise ValueError('must have at least one non-deprecated scheme')
        elif default in deps:
            raise ValueError('default scheme cannot be deprecated')
        for cat in self.categories:
            cdeps = dep_map.get(cat, deps)
            cdefault = default_map.get(cat, default)
            if not cdefault:
                for scheme in schemes:
                    if scheme not in cdeps:
                        default_map[cat] = scheme
                        break
                else:
                    raise ValueError('must have at least one non-deprecated scheme for %r category' % cat)
            elif cdefault in cdeps:
                raise ValueError('default scheme for %r category cannot be deprecated' % cat)

    def default_scheme(self, category):
        """return default scheme for specific category"""
        defaults = self._default_schemes
        try:
            return defaults[category]
        except KeyError:
            pass
        if not self.schemes:
            raise KeyError('no hash schemes configured for this CryptContext instance')
        return defaults[None]

    def is_deprecated_with_flag(self, scheme, category):
        """is scheme deprecated under particular category?"""
        depmap = self.get_context_optionmap('deprecated')

        def test(cat):
            source = depmap.get(cat, depmap.get(None))
            if source is None:
                return None
            elif 'auto' in source:
                return scheme != self.default_scheme(cat)
            else:
                return scheme in source
        value = test(None) or False
        if category:
            alt = test(category)
            if alt is not None and value != alt:
                return (alt, True)
        return (value, False)

    def _init_records(self):
        self._record_lists = {}
        records = self._records = {}
        all_context_kwds = self.context_kwds = set()
        get_options = self._get_record_options_with_flag
        categories = (None,) + self.categories
        for handler in self.handlers:
            scheme = handler.name
            all_context_kwds.update(handler.context_kwds)
            for cat in categories:
                kwds, has_cat_options = get_options(scheme, cat)
                if cat is None or has_cat_options:
                    records[scheme, cat] = self._create_record(handler, cat, **kwds)

    @staticmethod
    def _create_record(handler, category=None, deprecated=False, **settings):
        try:
            subcls = handler.using(relaxed=True, **settings)
        except TypeError as err:
            m = re.match(".* unexpected keyword argument '(.*)'$", str(err))
            if m and m.group(1) in settings:
                key = m.group(1)
                raise KeyError('keyword not supported by %s handler: %r' % (handler.name, key))
            raise
        assert subcls is not handler, 'expected unique variant of handler'
        subcls._Context__orig_handler = handler
        subcls.deprecated = deprecated
        return subcls

    def _get_record_options_with_flag(self, scheme, category):
        """return composite dict of options for given scheme + category.

        this is currently a private method, though some variant
        of its output may eventually be made public.

        given a scheme & category, it returns two things:
        a set of all the keyword options to pass to :meth:`_create_record`,
        and a bool flag indicating whether any of these options
        were specific to the named category. if this flag is false,
        the options are identical to the options for the default category.

        the options dict includes all the scheme-specific settings,
        as well as optional *deprecated* keyword.
        """
        kwds, has_cat_options = self.get_scheme_options_with_flag(scheme, category)
        value, not_inherited = self.is_deprecated_with_flag(scheme, category)
        if value:
            kwds['deprecated'] = True
        if not_inherited:
            has_cat_options = True
        return (kwds, has_cat_options)

    def get_record(self, scheme, category):
        """return record for specific scheme & category (cached)"""
        try:
            return self._records[scheme, category]
        except KeyError:
            pass
        if category is not None and (not isinstance(category, native_string_types)):
            if PY2 and isinstance(category, unicode):
                return self.get_record(scheme, category.encode('utf-8'))
            raise ExpectedTypeError(category, 'str or None', 'category')
        if scheme is not None and (not isinstance(scheme, native_string_types)):
            raise ExpectedTypeError(scheme, 'str or None', 'scheme')
        if not scheme:
            default = self.default_scheme(category)
            assert default
            record = self._records[None, category] = self.get_record(default, category)
            return record
        if category:
            try:
                cache = self._records
                record = cache[scheme, category] = cache[scheme, None]
                return record
            except KeyError:
                pass
        raise KeyError('crypt algorithm not found in policy: %r' % (scheme,))

    def _get_record_list(self, category=None):
        """return list of records for category (cached)

        this is an internal helper used only by identify_record()
        """
        try:
            return self._record_lists[category]
        except KeyError:
            pass
        value = self._record_lists[category] = [self.get_record(scheme, category) for scheme in self.schemes]
        return value

    def identify_record(self, hash, category, required=True):
        """internal helper to identify appropriate custom handler for hash"""
        if not isinstance(hash, unicode_or_bytes_types):
            raise ExpectedStringError(hash, 'hash')
        for record in self._get_record_list(category):
            if record.identify(hash):
                return record
        if not required:
            return None
        elif not self.schemes:
            raise KeyError('no crypt algorithms supported')
        else:
            raise exc.UnknownHashError('hash could not be identified')

    @memoized_property
    def disabled_record(self):
        for record in self._get_record_list(None):
            if record.is_disabled:
                return record
        raise RuntimeError("no disabled hasher present (perhaps add 'unix_disabled' to list of schemes?)")

    def iter_config(self, resolve=False):
        """regenerate original config.

        this is an iterator which yields ``(cat,scheme,option),value`` items,
        in the order they generally appear inside an INI file.
        if interpreted as a dictionary, it should match the original
        keywords passed to the CryptContext (aside from any canonization).

        it's mainly used as the internal backend for most of the public
        serialization methods.
        """
        scheme_options = self._scheme_options
        context_options = self._context_options
        scheme_keys = sorted(scheme_options)
        context_keys = sorted(context_options)
        if 'schemes' in context_keys:
            context_keys.remove('schemes')
        value = self.handlers if resolve else self.schemes
        if value:
            yield ((None, None, 'schemes'), list(value))
        for cat in (None,) + self.categories:
            for key in context_keys:
                try:
                    value = context_options[key][cat]
                except KeyError:
                    pass
                else:
                    if isinstance(value, list):
                        value = list(value)
                    yield ((cat, None, key), value)
            for scheme in scheme_keys:
                try:
                    kwds = scheme_options[scheme][cat]
                except KeyError:
                    pass
                else:
                    for key in sorted(kwds):
                        yield ((cat, scheme, key), kwds[key])