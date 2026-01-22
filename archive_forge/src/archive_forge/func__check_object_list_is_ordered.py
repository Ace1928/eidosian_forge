import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
def _check_object_list_is_ordered(self):
    """
        Warn if self.object_list is unordered (typically a QuerySet).
        """
    ordered = getattr(self.object_list, 'ordered', None)
    if ordered is not None and (not ordered):
        obj_list_repr = '{} {}'.format(self.object_list.model, self.object_list.__class__.__name__) if hasattr(self.object_list, 'model') else '{!r}'.format(self.object_list)
        warnings.warn('Pagination may yield inconsistent results with an unordered object_list: {}.'.format(obj_list_repr), UnorderedObjectListWarning, stacklevel=3)