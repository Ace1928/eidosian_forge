import re
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.transaction import atomic
from .exceptions import IrreversibleError
def unapply(self, project_state, schema_editor, collect_sql=False):
    """
        Take a project_state representing all migrations prior to this one
        and a schema_editor for a live database and apply the migration
        in a reverse order.

        The backwards migration process consists of two phases:

        1. The intermediate states from right before the first until right
           after the last operation inside this migration are preserved.
        2. The operations are applied in reverse order using the states
           recorded in step 1.
        """
    to_run = []
    new_state = project_state
    for operation in self.operations:
        if not operation.reversible:
            raise IrreversibleError('Operation %s in %s is not reversible' % (operation, self))
        new_state = new_state.clone()
        old_state = new_state.clone()
        operation.state_forwards(self.app_label, new_state)
        to_run.insert(0, (operation, old_state, new_state))
    for operation, to_state, from_state in to_run:
        if collect_sql:
            schema_editor.collected_sql.append('--')
            schema_editor.collected_sql.append('-- %s' % operation.describe())
            schema_editor.collected_sql.append('--')
            if not operation.reduces_to_sql:
                schema_editor.collected_sql.append('-- THIS OPERATION CANNOT BE WRITTEN AS SQL')
                continue
            collected_sql_before = len(schema_editor.collected_sql)
        atomic_operation = operation.atomic or (self.atomic and operation.atomic is not False)
        if not schema_editor.atomic_migration and atomic_operation:
            with atomic(schema_editor.connection.alias):
                operation.database_backwards(self.app_label, schema_editor, from_state, to_state)
        else:
            operation.database_backwards(self.app_label, schema_editor, from_state, to_state)
        if collect_sql and collected_sql_before == len(schema_editor.collected_sql):
            schema_editor.collected_sql.append('-- (no-op)')
    return project_state