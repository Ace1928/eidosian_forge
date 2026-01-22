import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
def _get_obj_path(self, obj):
    if isinstance(obj, models.LogBook):
        path = self.book_path
    elif isinstance(obj, models.FlowDetail):
        path = self.flow_path
    elif isinstance(obj, models.AtomDetail):
        path = self.atom_path
    else:
        raise exc.StorageFailure('Invalid storage class %s' % type(obj))
    return self._join_path(path, obj.uuid)