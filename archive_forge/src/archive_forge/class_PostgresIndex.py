from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class PostgresIndex(Index):

    @cached_property
    def max_name_length(self):
        return Index.max_name_length - len(Index.suffix) + len(self.suffix)

    def create_sql(self, model, schema_editor, using='', **kwargs):
        self.check_supported(schema_editor)
        statement = super().create_sql(model, schema_editor, using=' USING %s' % (using or self.suffix), **kwargs)
        with_params = self.get_with_params()
        if with_params:
            statement.parts['extra'] = ' WITH (%s)%s' % (', '.join(with_params), statement.parts['extra'])
        return statement

    def check_supported(self, schema_editor):
        pass

    def get_with_params(self):
        return []