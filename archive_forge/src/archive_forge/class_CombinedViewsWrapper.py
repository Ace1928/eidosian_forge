import logging
from os_ken.services.protocols.bgp.operator.views import fields
class CombinedViewsWrapper(RdyToFlattenList):
    """List-like wrapper for views. It provides same interface as any other
    views but enables as to access all views in bulk.
    It wraps and return responses from all views as a list. Be aware that
    in case of DictViews wrapped in CombinedViewsWrapper you loose
    information about dict keys.
    """

    def __init__(self, obj):
        super(CombinedViewsWrapper, self).__init__(obj)
        self._obj = obj

    def combine_related(self, field_name):
        return CombinedViewsWrapper(list(_flatten([obj.combine_related(field_name) for obj in self._obj])))

    def c_rel(self, *args, **kwargs):
        return self.combine_related(*args, **kwargs)

    def encode(self):
        return list(_flatten([obj.encode() for obj in self._obj]))

    def get_field(self, field_name):
        return list(_flatten([obj.get_field(field_name) for obj in self._obj]))

    @property
    def model(self):
        return list(_flatten([obj.model for obj in self._obj]))

    def apply_filter(self, filter_func):
        for obj in self._obj:
            obj.apply_filter(filter_func)

    def clear_filter(self):
        for obj in self._obj:
            obj.clear_filter()