from __future__ import annotations  # isort: split
import __future__  # Regular import, not special!
import enum
import functools
import importlib
import inspect
import json
import socket as stdlib_socket
import sys
import types
from pathlib import Path, PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
import attrs
import pytest
import trio
import trio.testing
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from .. import _core, _util
from .._core._tests.tutil import slow
from .pytest_plugin import RUN_SLOW
@slow
@pytest.mark.redistributors_should_skip()
@pytest.mark.skipif(sys.version_info.releaselevel == 'alpha', reason='skip static introspection tools on Python dev/alpha releases')
@pytest.mark.parametrize('module_name', PUBLIC_MODULE_NAMES)
@pytest.mark.parametrize('tool', ['jedi', 'mypy'])
def test_static_tool_sees_class_members(tool: str, module_name: str, tmp_path: Path) -> None:
    module = PUBLIC_MODULES[PUBLIC_MODULE_NAMES.index(module_name)]

    def no_hidden(symbols: Iterable[str]) -> set[str]:
        return {symbol for symbol in symbols if not symbol.startswith('_') or symbol.startswith('__')}
    if tool == 'mypy':
        if sys.implementation.name != 'cpython':
            pytest.skip('mypy not installed in tests on pypy')
        cache = Path.cwd() / '.mypy_cache'
        _ensure_mypy_cache_updated()
        trio_cache = next(cache.glob('*/trio'))
        modname = module_name
        _, modname = (modname + '.').split('.', 1)
        modname = modname[:-1]
        mod_cache = trio_cache / modname if modname else trio_cache
        if mod_cache.is_dir():
            mod_cache = mod_cache / '__init__.data.json'
        else:
            mod_cache = trio_cache / (modname + '.data.json')
        assert mod_cache.exists()
        assert mod_cache.is_file()
        with mod_cache.open() as cache_file:
            cache_json = json.loads(cache_file.read())

        @functools.lru_cache
        def lookup_symbol(symbol: str) -> dict[str, str]:
            topname, *modname, name = symbol.split('.')
            version = next(cache.glob('3.*/'))
            mod_cache = version / topname
            if not mod_cache.is_dir():
                mod_cache = version / (topname + '.data.json')
            if modname:
                for piece in modname[:-1]:
                    mod_cache /= piece
                next_cache = mod_cache / modname[-1]
                if next_cache.is_dir():
                    mod_cache = next_cache / '__init__.data.json'
                else:
                    mod_cache = mod_cache / (modname[-1] + '.data.json')
            elif mod_cache.is_dir():
                mod_cache /= '__init__.data.json'
            with mod_cache.open() as f:
                return json.loads(f.read())['names'][name]
    errors: dict[str, object] = {}
    for class_name, class_ in module.__dict__.items():
        if not isinstance(class_, type):
            continue
        if module_name == 'trio.socket' and class_name in dir(stdlib_socket):
            continue
        if class_ is trio.testing.RaisesGroup:
            continue
        ignore_names = set(dir(type(class_))) | {'__annotations__', '__attrs_attrs__', '__attrs_own_setattr__', '__callable_proto_members_only__', '__class_getitem__', '__final__', '__getstate__', '__match_args__', '__order__', '__orig_bases__', '__parameters__', '__protocol_attrs__', '__setstate__', '__slots__', '__weakref__', '__copy__', '__deepcopy__'}
        if sys.implementation.name == 'pypy':
            ignore_names |= {'__basicsize__', '__dictoffset__', '__itemsize__', '__sizeof__', '__weakrefoffset__', '__unicode__'}
        runtime_names = no_hidden((x[0] for x in inspect.getmembers(class_))) - ignore_names
        if tool == 'jedi':
            try:
                import jedi
            except ImportError as error:
                skip_if_optional_else_raise(error)
            script = jedi.Script(f'from {module_name} import {class_name}; {class_name}.')
            completions = script.complete()
            static_names = no_hidden((c.name for c in completions)) - ignore_names
        elif tool == 'mypy':
            cached_type_info = cache_json['names'][class_name]
            if 'node' not in cached_type_info:
                cached_type_info = lookup_symbol(cached_type_info['cross_ref'])
            assert 'node' in cached_type_info
            node = cached_type_info['node']
            static_names = no_hidden((k for k in node['names'] if not k.startswith('.')))
            for symbol in node['mro'][1:]:
                node = lookup_symbol(symbol)['node']
                static_names |= no_hidden((k for k in node['names'] if not k.startswith('.')))
            static_names -= ignore_names
        else:
            raise AssertionError('unknown tool')
        missing = runtime_names - static_names
        extra = static_names - runtime_names
        if tool == 'jedi' and BaseException in class_.__mro__ and (sys.version_info >= (3, 11)):
            missing.remove('add_note')
        if tool == 'mypy' and BaseException in class_.__mro__ and (sys.version_info >= (3, 11)):
            extra.remove('__notes__')
        if tool == 'mypy' and attrs.has(class_):
            before = len(extra)
            extra = {e for e in extra if not e.endswith('AttrsAttributes__')}
            assert len(extra) == before - 1
        if tool == 'mypy' and enum.Enum in class_.__mro__ and (sys.version_info >= (3, 12)):
            extra.remove('__signature__')
        if tool == 'mypy' and class_ == trio.Nursery:
            extra.remove('cancel_scope')
        EXTRAS = {trio.DTLSChannel: {'peer_address', 'endpoint'}, trio.DTLSEndpoint: {'socket', 'incoming_packets_buffer'}, trio.Process: {'args', 'pid', 'stderr', 'stdin', 'stdio', 'stdout'}, trio.SSLListener: {'transport_listener'}, trio.SSLStream: {'transport_stream'}, trio.SocketListener: {'socket'}, trio.SocketStream: {'socket'}, trio.testing.MemoryReceiveStream: {'close_hook', 'receive_some_hook'}, trio.testing.MemorySendStream: {'close_hook', 'send_all_hook', 'wait_send_all_might_not_block_hook'}, trio.testing.Matcher: {'exception_type', 'match', 'check'}}
        if tool == 'mypy' and class_ in EXTRAS:
            before = len(extra)
            extra -= EXTRAS[class_]
            assert len(extra) == before - len(EXTRAS[class_])
        if class_ == trio.StapledStream:
            extra.remove('receive_stream')
            extra.remove('send_stream')
        if tool == 'jedi' and sys.version_info >= (3, 11):
            if class_ in (trio.DTLSChannel, trio.MemoryReceiveChannel, trio.MemorySendChannel, trio.SSLListener, trio.SocketListener):
                missing.remove('__aenter__')
                missing.remove('__aexit__')
            if class_ in (trio.DTLSChannel, trio.MemoryReceiveChannel):
                missing.remove('__aiter__')
                missing.remove('__anext__')
        if class_ in (trio.Path, trio.WindowsPath, trio.PosixPath):
            missing -= PurePath.__dict__.keys()
            if tool == 'mypy' and sys.platform == 'win32':
                missing -= {'owner', 'is_mount', 'group'}
            if tool == 'jedi' and sys.platform == 'win32':
                extra -= {'owner', 'is_mount', 'group'}
        if missing or extra:
            errors[f'{module_name}.{class_name}'] = {'missing': missing, 'extra': extra}
    if errors:
        from pprint import pprint
        print(f"\n{tool} can't see the following symbols in {module_name}:")
        pprint(errors)
    assert not errors