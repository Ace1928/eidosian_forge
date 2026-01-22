import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def allows_group_by_selected_pks(self):
    if self.connection.mysql_is_mariadb:
        return 'ONLY_FULL_GROUP_BY' not in self.connection.sql_mode
    return True