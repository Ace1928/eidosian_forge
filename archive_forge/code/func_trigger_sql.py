from peewee import *
from playhouse.sqlite_ext import JSONField
def trigger_sql(self, model, action, skip_fields=None):
    assert action in self._actions
    use_old = action != 'INSERT'
    use_new = action != 'DELETE'
    cols = self._build_column_array(model, use_old, use_new, skip_fields)
    return self.template % {'table': model._meta.table_name, 'action': action, 'new_old': 'NEW' if action != 'DELETE' else 'OLD', 'primary_key': model._meta.primary_key.column_name, 'column_array': cols, 'change_table': self.table_name}