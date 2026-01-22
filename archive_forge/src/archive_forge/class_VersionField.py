import warnings
import django
from django.db import models
from . import base
class VersionField(SemVerField):
    default_error_messages = {'invalid': _('Enter a valid version number in X.Y.Z format.')}
    description = _('Version')

    def __init__(self, *args, **kwargs):
        self.partial = kwargs.pop('partial', False)
        if self.partial:
            warnings.warn('Use of `partial=True` will be removed in 3.0.', DeprecationWarning, stacklevel=2)
        self.coerce = kwargs.pop('coerce', False)
        super(VersionField, self).__init__(*args, **kwargs)

    def deconstruct(self):
        """Handle django.db.migrations."""
        name, path, args, kwargs = super(VersionField, self).deconstruct()
        kwargs['partial'] = self.partial
        kwargs['coerce'] = self.coerce
        return (name, path, args, kwargs)

    def to_python(self, value):
        """Converts any value to a base.Version field."""
        if value is None or value == '':
            return value
        if isinstance(value, base.Version):
            return value
        if self.coerce:
            return base.Version.coerce(value, partial=self.partial)
        else:
            return base.Version(value, partial=self.partial)