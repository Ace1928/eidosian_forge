from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class GinIndex(PostgresIndex):
    suffix = 'gin'

    def __init__(self, *expressions, fastupdate=None, gin_pending_list_limit=None, **kwargs):
        self.fastupdate = fastupdate
        self.gin_pending_list_limit = gin_pending_list_limit
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.fastupdate is not None:
            kwargs['fastupdate'] = self.fastupdate
        if self.gin_pending_list_limit is not None:
            kwargs['gin_pending_list_limit'] = self.gin_pending_list_limit
        return (path, args, kwargs)

    def get_with_params(self):
        with_params = []
        if self.gin_pending_list_limit is not None:
            with_params.append('gin_pending_list_limit = %d' % self.gin_pending_list_limit)
        if self.fastupdate is not None:
            with_params.append('fastupdate = %s' % ('on' if self.fastupdate else 'off'))
        return with_params