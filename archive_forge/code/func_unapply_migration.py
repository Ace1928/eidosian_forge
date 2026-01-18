from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def unapply_migration(self, state, migration, fake=False):
    """Run a migration backwards."""
    if self.progress_callback:
        self.progress_callback('unapply_start', migration, fake)
    if not fake:
        with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
            state = migration.unapply(state, schema_editor)
    if migration.replaces:
        for app_label, name in migration.replaces:
            self.recorder.record_unapplied(app_label, name)
    self.recorder.record_unapplied(migration.app_label, migration.name)
    if self.progress_callback:
        self.progress_callback('unapply_success', migration, fake)
    return state