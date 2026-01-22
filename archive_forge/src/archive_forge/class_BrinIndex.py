from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class BrinIndex(PostgresIndex):
    suffix = 'brin'

    def __init__(self, *expressions, autosummarize=None, pages_per_range=None, **kwargs):
        if pages_per_range is not None and pages_per_range <= 0:
            raise ValueError('pages_per_range must be None or a positive integer')
        self.autosummarize = autosummarize
        self.pages_per_range = pages_per_range
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.autosummarize is not None:
            kwargs['autosummarize'] = self.autosummarize
        if self.pages_per_range is not None:
            kwargs['pages_per_range'] = self.pages_per_range
        return (path, args, kwargs)

    def get_with_params(self):
        with_params = []
        if self.autosummarize is not None:
            with_params.append('autosummarize = %s' % ('on' if self.autosummarize else 'off'))
        if self.pages_per_range is not None:
            with_params.append('pages_per_range = %d' % self.pages_per_range)
        return with_params