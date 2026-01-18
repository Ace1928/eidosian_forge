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
@pytest.mark.redistributors_should_skip()
@pytest.mark.skipif(sys.version_info.releaselevel == 'alpha', reason='skip static introspection tools on Python dev/alpha releases')
@pytest.mark.parametrize('modname', PUBLIC_MODULE_NAMES)
@pytest.mark.parametrize('tool', ['pylint', 'jedi', 'mypy', 'pyright_verifytypes'])
@pytest.mark.filterwarnings("ignore:module 'sre_constants' is deprecated:DeprecationWarning")
def test_static_tool_sees_all_symbols(tool: str, modname: str, tmp_path: Path) -> None:
    module = importlib.import_module(modname)

    def no_underscores(symbols: Iterable[str]) -> set[str]:
        return {symbol for symbol in symbols if not symbol.startswith('_')}
    runtime_names = no_underscores(dir(module))
    if modname == 'trio':
        runtime_names.discard('tests')
    for name in __future__.all_feature_names:
        if getattr(module, name, None) is getattr(__future__, name):
            runtime_names.remove(name)
    if tool == 'pylint':
        try:
            from pylint.lint import PyLinter
        except ImportError as error:
            skip_if_optional_else_raise(error)
        linter = PyLinter()
        assert module.__file__ is not None
        ast = linter.get_ast(module.__file__, modname)
        static_names = no_underscores(ast)
    elif tool == 'jedi':
        if sys.implementation.name != 'cpython':
            pytest.skip('jedi does not support pypy')
        try:
            import jedi
        except ImportError as error:
            skip_if_optional_else_raise(error)
        script = jedi.Script(f'import {modname}; {modname}.')
        completions = script.complete()
        static_names = no_underscores((c.name for c in completions))
    elif tool == 'mypy':
        if not RUN_SLOW:
            pytest.skip('use --run-slow to check against mypy')
        if sys.implementation.name != 'cpython':
            pytest.skip('mypy not installed in tests on pypy')
        cache = Path.cwd() / '.mypy_cache'
        _ensure_mypy_cache_updated()
        trio_cache = next(cache.glob('*/trio'))
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
            static_names = no_underscores((key for key, value in cache_json['names'].items() if not key.startswith('.') and value['kind'] == 'Gdef'))
    elif tool == 'pyright_verifytypes':
        if not RUN_SLOW:
            pytest.skip('use --run-slow to check against pyright')
        try:
            import pyright
        except ImportError as error:
            skip_if_optional_else_raise(error)
        import subprocess
        res = subprocess.run(['pyright', f'--verifytypes={modname}', '--outputjson'], capture_output=True)
        current_result = json.loads(res.stdout)
        static_names = {x['name'][len(modname) + 1:] for x in current_result['typeCompleteness']['symbols'] if x['name'].startswith(modname)}
    else:
        raise AssertionError()
    missing_names = runtime_names - static_names
    missing_names -= {'tests'}
    if missing_names:
        print(f"{tool} can't see the following names in {modname}:")
        print()
        for name in sorted(missing_names):
            print(f'    {name}')
        raise AssertionError()