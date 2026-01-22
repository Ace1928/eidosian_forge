from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
class MutateTestCase(BaseTestCase):

    def setUp(self):
        super(MutateTestCase, self).setUp()
        if hasattr(log._load_log_config, 'old_time'):
            del log._load_log_config.old_time

    def setup_confs(self, *confs):
        paths = self.create_tempfiles((('conf_%d' % i, conf) for i, conf in enumerate(confs)))
        self.CONF(['--config-file', paths[0]])
        return paths

    def test_debug(self):
        paths = self.setup_confs('[DEFAULT]\ndebug = false\n', '[DEFAULT]\ndebug = true\n')
        log_root = log.getLogger(None).logger
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        self.assertEqual(False, self.CONF.debug)
        self.assertEqual(log.INFO, log_root.getEffectiveLevel())
        shutil.copy(paths[1], paths[0])
        self.CONF.mutate_config_files()
        self.assertEqual(True, self.CONF.debug)
        self.assertEqual(log.DEBUG, log_root.getEffectiveLevel())

    @mock.patch.object(logging.config, 'fileConfig')
    def test_log_config_append(self, mock_fileConfig):
        logini = self.create_tempfiles([('log.ini', MIN_LOG_INI)])[0]
        paths = self.setup_confs('[DEFAULT]\nlog_config_append = no_exist\n', '[DEFAULT]\nlog_config_append = %s\n' % logini)
        self.assertRaises(log.LogConfigError, log.setup, self.CONF, '')
        self.assertFalse(mock_fileConfig.called)
        shutil.copy(paths[1], paths[0])
        self.CONF.mutate_config_files()
        mock_fileConfig.assert_called_once_with(logini, disable_existing_loggers=False)

    @mock.patch.object(logging.config, 'fileConfig')
    def test_log_config_append_no_touch(self, mock_fileConfig):
        logini = self.create_tempfiles([('log.ini', MIN_LOG_INI)])[0]
        self.setup_confs('[DEFAULT]\nlog_config_append = %s\n' % logini)
        log.setup(self.CONF, '')
        mock_fileConfig.assert_called_once_with(logini, disable_existing_loggers=False)
        mock_fileConfig.reset_mock()
        self.CONF.mutate_config_files()
        self.assertFalse(mock_fileConfig.called)

    @mock.patch.object(logging.config, 'fileConfig')
    def test_log_config_append_touch(self, mock_fileConfig):
        logini = self.create_tempfiles([('log.ini', MIN_LOG_INI)])[0]
        self.setup_confs('[DEFAULT]\nlog_config_append = %s\n' % logini)
        log.setup(self.CONF, '')
        mock_fileConfig.assert_called_once_with(logini, disable_existing_loggers=False)
        mock_fileConfig.reset_mock()
        time.sleep(1)
        os.utime(logini, None)
        self.CONF.mutate_config_files()
        mock_fileConfig.assert_called_once_with(logini, disable_existing_loggers=False)

    def mk_log_config(self, data):
        """Turns a dictConfig-like structure into one suitable for fileConfig.

        The schema is not validated as this is a test helper not production
        code. Garbage in, garbage out. Particularly, don't try to use filters,
        fileConfig doesn't support them.

        Handler args must be passed like 'args': (1, 2). dictConfig passes
        keys by keyword name and fileConfig passes them by position so
        accepting the dictConfig form makes it nigh impossible to produce the
        fileConfig form.

        I traverse dicts by sorted keys for output stability but it doesn't
        matter if defaulted keys are out of order.
        """
        lines = []
        for section in ['formatters', 'handlers', 'loggers']:
            items = data.get(section, {})
            keys = sorted(items)
            skeys = ','.join(keys)
            if section == 'loggers' and 'root' in data:
                skeys = 'root,' + skeys if skeys else 'root'
            lines.extend(['[%s]' % section, 'keys=%s' % skeys])
            for key in keys:
                lines.extend(['', '[%s_%s]' % (section[:-1], key)])
                item = items[key]
                lines.extend(('%s=%s' % (k, item[k]) for k in sorted(item)))
                if section == 'handlers':
                    if 'args' not in item:
                        lines.append('args=()')
                elif section == 'loggers':
                    lines.append('qualname=%s' % key)
                    if 'handlers' not in item:
                        lines.append('handlers=')
            lines.append('')
        root = data.get('root', {})
        if root:
            lines.extend(['[logger_root]'])
            lines.extend(('%s=%s' % (k, root[k]) for k in sorted(root)))
            if 'handlers' not in root:
                lines.append('handlers=')
        return '\n'.join(lines)

    def test_mk_log_config_full(self):
        data = {'loggers': {'aaa': {'level': 'INFO'}, 'bbb': {'level': 'WARN', 'propagate': False}}, 'handlers': {'aaa': {'level': 'INFO'}, 'bbb': {'level': 'WARN', 'propagate': False, 'args': (1, 2)}}, 'formatters': {'aaa': {'level': 'INFO'}, 'bbb': {'level': 'WARN', 'propagate': False}}, 'root': {'level': 'INFO', 'handlers': 'aaa'}}
        full = '[formatters]\nkeys=aaa,bbb\n\n[formatter_aaa]\nlevel=INFO\n\n[formatter_bbb]\nlevel=WARN\npropagate=False\n\n[handlers]\nkeys=aaa,bbb\n\n[handler_aaa]\nlevel=INFO\nargs=()\n\n[handler_bbb]\nargs=(1, 2)\nlevel=WARN\npropagate=False\n\n[loggers]\nkeys=root,aaa,bbb\n\n[logger_aaa]\nlevel=INFO\nqualname=aaa\nhandlers=\n\n[logger_bbb]\nlevel=WARN\npropagate=False\nqualname=bbb\nhandlers=\n\n[logger_root]\nhandlers=aaa\nlevel=INFO'
        self.assertEqual(full, self.mk_log_config(data))

    def test_mk_log_config_empty(self):
        """Ensure mk_log_config tolerates missing bits"""
        empty = '[formatters]\nkeys=\n\n[handlers]\nkeys=\n\n[loggers]\nkeys=\n'
        self.assertEqual(empty, self.mk_log_config({}))

    @contextmanager
    def mutate_conf(self, conf1, conf2):
        loginis = self.create_tempfiles([('log1.ini', self.mk_log_config(conf1)), ('log2.ini', self.mk_log_config(conf2))])
        confs = self.setup_confs('[DEFAULT]\nlog_config_append = %s\n' % loginis[0], '[DEFAULT]\nlog_config_append = %s\n' % loginis[1])
        log.setup(self.CONF, '')
        yield (loginis, confs)
        shutil.copy(confs[1], confs[0])
        os.utime(self.CONF.log_config_append, (0, 0))
        self.CONF.mutate_config_files()

    @mock.patch.object(logging.config, 'fileConfig')
    def test_log_config_append_change_file(self, mock_fileConfig):
        with self.mutate_conf({}, {}) as (loginis, confs):
            mock_fileConfig.assert_called_once_with(loginis[0], disable_existing_loggers=False)
            mock_fileConfig.reset_mock()
        mock_fileConfig.assert_called_once_with(loginis[1], disable_existing_loggers=False)

    def set_root_stream(self):
        root = logging.getLogger()
        self.assertEqual(1, len(root.handlers))
        handler = root.handlers[0]
        handler.stream = io.StringIO()
        return handler.stream

    def test_remove_handler(self):
        fake_handler = {'class': 'logging.StreamHandler', 'args': ()}
        conf1 = {'root': {'handlers': 'fake'}, 'handlers': {'fake': fake_handler}}
        conf2 = {'root': {'handlers': ''}}
        with self.mutate_conf(conf1, conf2) as (loginis, confs):
            stream = self.set_root_stream()
            root = logging.getLogger()
            root.error('boo')
            self.assertEqual('boo\n', stream.getvalue())
        stream.truncate(0)
        root.error('boo')
        self.assertEqual('', stream.getvalue())

    def test_remove_logger(self):
        fake_handler = {'class': 'logging.StreamHandler'}
        fake_logger = {'level': 'WARN'}
        conf1 = {'root': {'handlers': 'fake'}, 'handlers': {'fake': fake_handler}, 'loggers': {'a.a': fake_logger}}
        conf2 = {'root': {'handlers': 'fake'}, 'handlers': {'fake': fake_handler}}
        stream = io.StringIO()
        with self.mutate_conf(conf1, conf2) as (loginis, confs):
            stream = self.set_root_stream()
            log = logging.getLogger('a.a')
            log.info('info')
            log.warn('warn')
            self.assertEqual('warn\n', stream.getvalue())
        stream = self.set_root_stream()
        log.info('info')
        log.warn('warn')
        self.assertEqual('info\nwarn\n', stream.getvalue())