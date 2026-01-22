import os
import re
from importlib import import_module
from django import get_version
from django.apps import apps
from django.conf import SettingsReference  # NOQA
from django.db import migrations
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.serializer import Serializer, serializer_factory
from django.utils.inspect import get_func_args
from django.utils.module_loading import module_dir
from django.utils.timezone import now
class OperationWriter:

    def __init__(self, operation, indentation=2):
        self.operation = operation
        self.buff = []
        self.indentation = indentation

    def serialize(self):

        def _write(_arg_name, _arg_value):
            if _arg_name in self.operation.serialization_expand_args and isinstance(_arg_value, (list, tuple, dict)):
                if isinstance(_arg_value, dict):
                    self.feed('%s={' % _arg_name)
                    self.indent()
                    for key, value in _arg_value.items():
                        key_string, key_imports = MigrationWriter.serialize(key)
                        arg_string, arg_imports = MigrationWriter.serialize(value)
                        args = arg_string.splitlines()
                        if len(args) > 1:
                            self.feed('%s: %s' % (key_string, args[0]))
                            for arg in args[1:-1]:
                                self.feed(arg)
                            self.feed('%s,' % args[-1])
                        else:
                            self.feed('%s: %s,' % (key_string, arg_string))
                        imports.update(key_imports)
                        imports.update(arg_imports)
                    self.unindent()
                    self.feed('},')
                else:
                    self.feed('%s=[' % _arg_name)
                    self.indent()
                    for item in _arg_value:
                        arg_string, arg_imports = MigrationWriter.serialize(item)
                        args = arg_string.splitlines()
                        if len(args) > 1:
                            for arg in args[:-1]:
                                self.feed(arg)
                            self.feed('%s,' % args[-1])
                        else:
                            self.feed('%s,' % arg_string)
                        imports.update(arg_imports)
                    self.unindent()
                    self.feed('],')
            else:
                arg_string, arg_imports = MigrationWriter.serialize(_arg_value)
                args = arg_string.splitlines()
                if len(args) > 1:
                    self.feed('%s=%s' % (_arg_name, args[0]))
                    for arg in args[1:-1]:
                        self.feed(arg)
                    self.feed('%s,' % args[-1])
                else:
                    self.feed('%s=%s,' % (_arg_name, arg_string))
                imports.update(arg_imports)
        imports = set()
        name, args, kwargs = self.operation.deconstruct()
        operation_args = get_func_args(self.operation.__init__)
        if getattr(migrations, name, None) == self.operation.__class__:
            self.feed('migrations.%s(' % name)
        else:
            imports.add('import %s' % self.operation.__class__.__module__)
            self.feed('%s.%s(' % (self.operation.__class__.__module__, name))
        self.indent()
        for i, arg in enumerate(args):
            arg_value = arg
            arg_name = operation_args[i]
            _write(arg_name, arg_value)
        i = len(args)
        for arg_name in operation_args[i:]:
            if arg_name in kwargs:
                arg_value = kwargs[arg_name]
                _write(arg_name, arg_value)
        self.unindent()
        self.feed('),')
        return (self.render(), imports)

    def indent(self):
        self.indentation += 1

    def unindent(self):
        self.indentation -= 1

    def feed(self, line):
        self.buff.append(' ' * (self.indentation * 4) + line)

    def render(self):
        return '\n'.join(self.buff)