from __future__ import annotations
import collections, functools, importlib
import typing as T
from .base import ExternalDependency, DependencyException, DependencyMethods, NotFoundDependency
from ..mesonlib import listify, MachineChoice, PerMachine
from .. import mlog
def _build_external_dependency_list(name: str, env: 'Environment', for_machine: MachineChoice, kwargs: T.Dict[str, T.Any]) -> T.List['DependencyGenerator']:
    if 'method' in kwargs and kwargs['method'] not in [e.value for e in DependencyMethods]:
        raise DependencyException('method {!r} is invalid'.format(kwargs['method']))
    lname = name.lower()
    if lname in packages:
        if isinstance(packages[lname], type):
            entry1 = T.cast('T.Type[ExternalDependency]', packages[lname])
            if issubclass(entry1, ExternalDependency):
                func: T.Callable[[], 'ExternalDependency'] = functools.partial(entry1, env, kwargs)
                dep = [func]
        else:
            entry2 = T.cast('T.Union[DependencyFactory, WrappedFactoryFunc]', packages[lname])
            dep = entry2(env, for_machine, kwargs)
        return dep
    candidates: T.List['DependencyGenerator'] = []
    if kwargs.get('method', 'auto') == 'auto':
        methods = ['pkg-config', 'extraframework', 'cmake']
    else:
        methods = [kwargs['method']]
    if 'dub' in methods:
        from .dub import DubDependency
        candidates.append(functools.partial(DubDependency, name, env, kwargs))
    if 'pkg-config' in methods:
        from .pkgconfig import PkgConfigDependency
        candidates.append(functools.partial(PkgConfigDependency, name, env, kwargs))
    if 'extraframework' in methods:
        if env.machines[for_machine].is_darwin():
            from .framework import ExtraFrameworkDependency
            candidates.append(functools.partial(ExtraFrameworkDependency, name, env, kwargs))
    if 'cmake' in methods:
        from .cmake import CMakeDependency
        candidates.append(functools.partial(CMakeDependency, name, env, kwargs))
    return candidates