import warnings
from django.db.models import CharField, EmailField, TextField
from django.test.utils import ignore_warnings
from django.utils.deprecation import RemovedInDjango51Warning
class CIText:

    def __init__(self, *args, **kwargs):
        warnings.warn('django.contrib.postgres.fields.CIText mixin is deprecated.', RemovedInDjango51Warning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def get_internal_type(self):
        return 'CI' + super().get_internal_type()

    def db_type(self, connection):
        return 'citext'