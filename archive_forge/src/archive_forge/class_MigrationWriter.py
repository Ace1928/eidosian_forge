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
class MigrationWriter:
    """
    Take a Migration instance and is able to produce the contents
    of the migration file from it.
    """

    def __init__(self, migration, include_header=True):
        self.migration = migration
        self.include_header = include_header
        self.needs_manual_porting = False

    def as_string(self):
        """Return a string of the file contents."""
        items = {'replaces_str': '', 'initial_str': ''}
        imports = set()
        operations = []
        for operation in self.migration.operations:
            operation_string, operation_imports = OperationWriter(operation).serialize()
            imports.update(operation_imports)
            operations.append(operation_string)
        items['operations'] = '\n'.join(operations) + '\n' if operations else ''
        dependencies = []
        for dependency in self.migration.dependencies:
            if dependency[0] == '__setting__':
                dependencies.append('        migrations.swappable_dependency(settings.%s),' % dependency[1])
                imports.add('from django.conf import settings')
            else:
                dependencies.append('        %s,' % self.serialize(dependency)[0])
        items['dependencies'] = '\n'.join(sorted(dependencies)) + '\n' if dependencies else ''
        migration_imports = set()
        for line in list(imports):
            if re.match('^import (.*)\\.\\d+[^\\s]*$', line):
                migration_imports.add(line.split('import')[1].strip())
                imports.remove(line)
                self.needs_manual_porting = True
        if 'from django.db import models' in imports:
            imports.discard('from django.db import models')
            imports.add('from django.db import migrations, models')
        else:
            imports.add('from django.db import migrations')
        sorted_imports = sorted(imports, key=lambda i: (i.split()[0] == 'from', i.split()[1]))
        items['imports'] = '\n'.join(sorted_imports) + '\n' if imports else ''
        if migration_imports:
            items['imports'] += '\n\n# Functions from the following migrations need manual copying.\n# Move them and any dependencies into this file, then update the\n# RunPython operations to refer to the local versions:\n# %s' % '\n# '.join(sorted(migration_imports))
        if self.migration.replaces:
            items['replaces_str'] = '\n    replaces = %s\n' % self.serialize(self.migration.replaces)[0]
        if self.include_header:
            items['migration_header'] = MIGRATION_HEADER_TEMPLATE % {'version': get_version(), 'timestamp': now().strftime('%Y-%m-%d %H:%M')}
        else:
            items['migration_header'] = ''
        if self.migration.initial:
            items['initial_str'] = '\n    initial = True\n'
        return MIGRATION_TEMPLATE % items

    @property
    def basedir(self):
        migrations_package_name, _ = MigrationLoader.migrations_module(self.migration.app_label)
        if migrations_package_name is None:
            raise ValueError("Django can't create migrations for app '%s' because migrations have been disabled via the MIGRATION_MODULES setting." % self.migration.app_label)
        try:
            migrations_module = import_module(migrations_package_name)
        except ImportError:
            pass
        else:
            try:
                return module_dir(migrations_module)
            except ValueError:
                pass
        app_config = apps.get_app_config(self.migration.app_label)
        maybe_app_name, _, migrations_package_basename = migrations_package_name.rpartition('.')
        if app_config.name == maybe_app_name:
            return os.path.join(app_config.path, migrations_package_basename)
        existing_dirs, missing_dirs = (migrations_package_name.split('.'), [])
        while existing_dirs:
            missing_dirs.insert(0, existing_dirs.pop(-1))
            try:
                base_module = import_module('.'.join(existing_dirs))
            except (ImportError, ValueError):
                continue
            else:
                try:
                    base_dir = module_dir(base_module)
                except ValueError:
                    continue
                else:
                    break
        else:
            raise ValueError('Could not locate an appropriate location to create migrations package %s. Make sure the toplevel package exists and can be imported.' % migrations_package_name)
        final_dir = os.path.join(base_dir, *missing_dirs)
        os.makedirs(final_dir, exist_ok=True)
        for missing_dir in missing_dirs:
            base_dir = os.path.join(base_dir, missing_dir)
            with open(os.path.join(base_dir, '__init__.py'), 'w'):
                pass
        return final_dir

    @property
    def filename(self):
        return '%s.py' % self.migration.name

    @property
    def path(self):
        return os.path.join(self.basedir, self.filename)

    @classmethod
    def serialize(cls, value):
        return serializer_factory(value).serialize()

    @classmethod
    def register_serializer(cls, type_, serializer):
        Serializer.register(type_, serializer)

    @classmethod
    def unregister_serializer(cls, type_):
        Serializer.unregister(type_)