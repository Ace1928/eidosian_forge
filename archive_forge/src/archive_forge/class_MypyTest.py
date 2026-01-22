from __future__ import annotations
import inspect
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from .base import TestBase
from .. import config
from ..assertions import eq_
from ... import util
@config.add_to_marker.mypy
class MypyTest(TestBase):
    __requires__ = ('no_sqlalchemy2_stubs',)

    @config.fixture(scope='function')
    def per_func_cachedir(self):
        yield from self._cachedir()

    @config.fixture(scope='class')
    def cachedir(self):
        yield from self._cachedir()

    def _cachedir(self):
        mypy_path = ''
        with tempfile.TemporaryDirectory() as cachedir:
            with open(Path(cachedir) / 'sqla_mypy_config.cfg', 'w') as config_file:
                config_file.write(f'\n                    [mypy]\n\n                    plugins = sqlalchemy.ext.mypy.plugin\n\n                    show_error_codes = True\n\n                    {mypy_path}\n                    disable_error_code = no-untyped-call\n\n                    [mypy-sqlalchemy.*]\n                    ignore_errors = True\n\n                    ')
            with open(Path(cachedir) / 'plain_mypy_config.cfg', 'w') as config_file:
                config_file.write(f'\n                    [mypy]\n\n                    show_error_codes = True\n\n                    {mypy_path}\n                    disable_error_code = var-annotated,no-untyped-call\n                    [mypy-sqlalchemy.*]\n                    ignore_errors = True\n\n                    ')
            yield cachedir

    @config.fixture()
    def mypy_runner(self, cachedir):
        from mypy import api

        def run(path, use_plugin=False, use_cachedir=None):
            if use_cachedir is None:
                use_cachedir = cachedir
            args = ['--strict', '--raise-exceptions', '--cache-dir', use_cachedir, '--config-file', os.path.join(use_cachedir, 'sqla_mypy_config.cfg' if use_plugin else 'plain_mypy_config.cfg')]
            filename = os.path.basename(path)
            test_program = os.path.join(use_cachedir, filename)
            if path != test_program:
                shutil.copyfile(path, test_program)
            args.append(test_program)
            os.environ.pop('MYPY_FORCE_COLOR', None)
            stdout, stderr, exitcode = api.run(args)
            return (stdout, stderr, exitcode)
        return run

    @config.fixture
    def mypy_typecheck_file(self, mypy_runner):

        def run(path, use_plugin=False):
            expected_messages = self._collect_messages(path)
            stdout, stderr, exitcode = mypy_runner(path, use_plugin=use_plugin)
            self._check_output(path, expected_messages, stdout, stderr, exitcode)
        return run

    @staticmethod
    def file_combinations(dirname):
        if os.path.isabs(dirname):
            path = dirname
        else:
            caller_path = inspect.stack()[1].filename
            path = os.path.join(os.path.dirname(caller_path), dirname)
        files = list(Path(path).glob('**/*.py'))
        for extra_dir in config.options.mypy_extra_test_paths:
            if extra_dir and os.path.isdir(extra_dir):
                files.extend((Path(extra_dir) / dirname).glob('**/*.py'))
        return files

    def _collect_messages(self, path):
        from sqlalchemy.ext.mypy.util import mypy_14
        expected_messages = []
        expected_re = re.compile('\\s*# EXPECTED(_MYPY)?(_RE)?(_TYPE)?: (.+)')
        py_ver_re = re.compile('^#\\s*PYTHON_VERSION\\s?>=\\s?(\\d+\\.\\d+)')
        with open(path) as file_:
            current_assert_messages = []
            for num, line in enumerate(file_, 1):
                m = py_ver_re.match(line)
                if m:
                    major, _, minor = m.group(1).partition('.')
                    if sys.version_info < (int(major), int(minor)):
                        config.skip_test('Requires python >= %s' % m.group(1))
                    continue
                m = expected_re.match(line)
                if m:
                    is_mypy = bool(m.group(1))
                    is_re = bool(m.group(2))
                    is_type = bool(m.group(3))
                    expected_msg = re.sub('# noqa[:]? ?.*', '', m.group(4))
                    if is_type:
                        if not is_re:
                            expected_msg = re.sub('([\\[\\]])', lambda m: f'\\{m.group(0)}', expected_msg)
                            expected_msg = re.sub('([\\w_]+)', lambda m: f'(?:.*\\.)?{m.group(1)}\\*?', expected_msg)
                            expected_msg = re.sub('List', 'builtins.list', expected_msg)
                            expected_msg = re.sub('\\b(int|str|float|bool)\\b', lambda m: f'builtins.{m.group(0)}\\*?', expected_msg)
                        is_mypy = is_re = True
                        expected_msg = f'Revealed type is "{expected_msg}"'
                    if mypy_14 and util.py39:
                        expected_msg = expected_msg[0] + re.sub('\\b(List|Tuple|Dict|Set)\\b' if is_type else '\\b(List|Tuple|Dict|Set|Type)\\b', lambda m: m.group(1).lower(), expected_msg[1:])
                    if mypy_14 and util.py310:
                        expected_msg = re.sub('Optional\\[(.*?)\\]', lambda m: f'{m.group(1)} | None', expected_msg)
                    current_assert_messages.append((is_mypy, is_re, expected_msg.strip()))
                elif current_assert_messages:
                    expected_messages.extend(((num, is_mypy, is_re, expected_msg) for is_mypy, is_re, expected_msg in current_assert_messages))
                    current_assert_messages[:] = []
        return expected_messages

    def _check_output(self, path, expected_messages, stdout, stderr, exitcode):
        not_located = []
        filename = os.path.basename(path)
        if expected_messages:
            output = []
            raw_lines = stdout.split('\n')
            while raw_lines:
                e = raw_lines.pop(0)
                if re.match('.+\\.py:\\d+: error: .*', e):
                    output.append(('error', e))
                elif re.match('.+\\.py:\\d+: note: +(?:Possible overload|def ).*', e):
                    while raw_lines:
                        ol = raw_lines.pop(0)
                        if not re.match('.+\\.py:\\d+: note: +def \\[.*', ol):
                            break
                elif re.match('.+\\.py:\\d+: note: .*(?:perhaps|suggestion)', e, re.I):
                    pass
                elif re.match('.+\\.py:\\d+: note: .*', e):
                    output.append(('note', e))
            for num, is_mypy, is_re, msg in expected_messages:
                msg = msg.replace("'", '"')
                prefix = '[SQLAlchemy Mypy plugin] ' if not is_mypy else ''
                for idx, (typ, errmsg) in enumerate(output):
                    if is_re:
                        if re.match(f'.*{filename}\\:{num}\\: {typ}\\: {prefix}{msg}', errmsg):
                            break
                    elif f'{filename}:{num}: {typ}: {prefix}{msg}' in errmsg.replace("'", '"'):
                        break
                else:
                    not_located.append(msg)
                    continue
                del output[idx]
            if not_located:
                missing = '\n'.join(not_located)
                print("Couldn't locate expected messages:", missing, sep='\n')
                if output:
                    extra = '\n'.join((msg for _, msg in output))
                    print('Remaining messages:', extra, sep='\n')
                assert False, 'expected messages not found, see stdout'
            if output:
                print(f'{len(output)} messages from mypy were not consumed:')
                print('\n'.join((msg for _, msg in output)))
                assert False, 'errors and/or notes remain, see stdout'
        else:
            if exitcode != 0:
                print(stdout, stderr, sep='\n')
            eq_(exitcode, 0, msg=stdout)