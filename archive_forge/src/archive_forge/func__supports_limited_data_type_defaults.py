from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
@property
def _supports_limited_data_type_defaults(self):
    if self.connection.mysql_is_mariadb:
        return True
    return self.connection.mysql_version >= (8, 0, 13)