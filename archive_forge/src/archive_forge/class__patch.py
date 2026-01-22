from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class _patch(object):
    attribute_name = None
    _active_patches = []

    def __init__(self, getter, attribute, new, spec, create, spec_set, autospec, new_callable, kwargs):
        if new_callable is not None:
            if new is not DEFAULT:
                raise ValueError("Cannot use 'new' and 'new_callable' together")
            if autospec is not None:
                raise ValueError("Cannot use 'autospec' and 'new_callable' together")
        self.getter = getter
        self.attribute = attribute
        self.new = new
        self.new_callable = new_callable
        self.spec = spec
        self.create = create
        self.has_local = False
        self.spec_set = spec_set
        self.autospec = autospec
        self.kwargs = kwargs
        self.additional_patchers = []

    def copy(self):
        patcher = _patch(self.getter, self.attribute, self.new, self.spec, self.create, self.spec_set, self.autospec, self.new_callable, self.kwargs)
        patcher.attribute_name = self.attribute_name
        patcher.additional_patchers = [p.copy() for p in self.additional_patchers]
        return patcher

    def __call__(self, func):
        if isinstance(func, ClassTypes):
            return self.decorate_class(func)
        return self.decorate_callable(func)

    def decorate_class(self, klass):
        for attr in dir(klass):
            if not attr.startswith(patch.TEST_PREFIX):
                continue
            attr_value = getattr(klass, attr)
            if not hasattr(attr_value, '__call__'):
                continue
            patcher = self.copy()
            setattr(klass, attr, patcher(attr_value))
        return klass

    def decorate_callable(self, func):
        if hasattr(func, 'patchings'):
            func.patchings.append(self)
            return func

        @wraps(func)
        def patched(*args, **keywargs):
            extra_args = []
            entered_patchers = []
            exc_info = tuple()
            try:
                for patching in patched.patchings:
                    arg = patching.__enter__()
                    entered_patchers.append(patching)
                    if patching.attribute_name is not None:
                        keywargs.update(arg)
                    elif patching.new is DEFAULT:
                        extra_args.append(arg)
                args += tuple(extra_args)
                return func(*args, **keywargs)
            except:
                if patching not in entered_patchers and _is_started(patching):
                    entered_patchers.append(patching)
                exc_info = sys.exc_info()
                raise
            finally:
                for patching in reversed(entered_patchers):
                    patching.__exit__(*exc_info)
        patched.patchings = [self]
        return patched

    def get_original(self):
        target = self.getter()
        name = self.attribute
        original = DEFAULT
        local = False
        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True
        if name in _builtins and isinstance(target, ModuleType):
            self.create = True
        if not self.create and original is DEFAULT:
            raise AttributeError('%s does not have the attribute %r' % (target, name))
        return (original, local)

    def __enter__(self):
        """Perform the patch."""
        new, spec, spec_set = (self.new, self.spec, self.spec_set)
        autospec, kwargs = (self.autospec, self.kwargs)
        new_callable = self.new_callable
        self.target = self.getter()
        if spec is False:
            spec = None
        if spec_set is False:
            spec_set = None
        if autospec is False:
            autospec = None
        if spec is not None and autospec is not None:
            raise TypeError("Can't specify spec and autospec")
        if (spec is not None or autospec is not None) and spec_set not in (True, None):
            raise TypeError("Can't provide explicit spec_set *and* spec or autospec")
        original, local = self.get_original()
        if new is DEFAULT and autospec is None:
            inherit = False
            if spec is True:
                spec = original
                if spec_set is True:
                    spec_set = original
                    spec = None
            elif spec is not None:
                if spec_set is True:
                    spec_set = spec
                    spec = None
            elif spec_set is True:
                spec_set = original
            if spec is not None or spec_set is not None:
                if original is DEFAULT:
                    raise TypeError("Can't use 'spec' with create=True")
                if isinstance(original, ClassTypes):
                    inherit = True
            Klass = MagicMock
            _kwargs = {}
            if new_callable is not None:
                Klass = new_callable
            elif spec is not None or spec_set is not None:
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if _is_list(this_spec):
                    not_callable = '__call__' not in this_spec
                else:
                    not_callable = not _callable(this_spec)
                if not_callable:
                    Klass = NonCallableMagicMock
            if spec is not None:
                _kwargs['spec'] = spec
            if spec_set is not None:
                _kwargs['spec_set'] = spec_set
            if isinstance(Klass, type) and issubclass(Klass, NonCallableMock) and self.attribute:
                _kwargs['name'] = self.attribute
            _kwargs.update(kwargs)
            new = Klass(**_kwargs)
            if inherit and _is_instance_mock(new):
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if not _is_list(this_spec) and (not _instance_callable(this_spec)):
                    Klass = NonCallableMagicMock
                _kwargs.pop('name')
                new.return_value = Klass(_new_parent=new, _new_name='()', **_kwargs)
        elif autospec is not None:
            if new is not DEFAULT:
                raise TypeError("autospec creates the mock for you. Can't specify autospec and new.")
            if original is DEFAULT:
                raise TypeError("Can't use 'autospec' with create=True")
            spec_set = bool(spec_set)
            if autospec is True:
                autospec = original
            new = create_autospec(autospec, spec_set=spec_set, _name=self.attribute, **kwargs)
        elif kwargs:
            raise TypeError("Can't pass kwargs to a mock we aren't creating")
        new_attr = new
        self.temp_original = original
        self.is_local = local
        setattr(self.target, self.attribute, new_attr)
        if self.attribute_name is not None:
            extra_args = {}
            if self.new is DEFAULT:
                extra_args[self.attribute_name] = new
            for patching in self.additional_patchers:
                arg = patching.__enter__()
                if patching.new is DEFAULT:
                    extra_args.update(arg)
            return extra_args
        return new

    def __exit__(self, *exc_info):
        """Undo the patch."""
        if not _is_started(self):
            raise RuntimeError('stop called on unstarted patcher')
        if self.is_local and self.temp_original is not DEFAULT:
            setattr(self.target, self.attribute, self.temp_original)
        else:
            delattr(self.target, self.attribute)
            if not self.create and (not hasattr(self.target, self.attribute) or self.attribute in ('__doc__', '__module__', '__defaults__', '__annotations__', '__kwdefaults__')):
                setattr(self.target, self.attribute, self.temp_original)
        del self.temp_original
        del self.is_local
        del self.target
        for patcher in reversed(self.additional_patchers):
            if _is_started(patcher):
                patcher.__exit__(*exc_info)

    def start(self):
        """Activate a patch, returning any created mock."""
        result = self.__enter__()
        self._active_patches.append(self)
        return result

    def stop(self):
        """Stop an active patch."""
        try:
            self._active_patches.remove(self)
        except ValueError:
            pass
        return self.__exit__()