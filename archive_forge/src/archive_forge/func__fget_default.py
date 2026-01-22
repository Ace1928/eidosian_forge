from .. import inspect
from ..ext.hybrid import hybrid_property
from ..orm.attributes import flag_modified
def _fget_default(self, err=None):
    if self.default == self._NO_DEFAULT_ARGUMENT:
        raise AttributeError(self.attr_name) from err
    else:
        return self.default