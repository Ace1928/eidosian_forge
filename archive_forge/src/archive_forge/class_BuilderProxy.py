from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
class BuilderProxy(builder):

    def __init__(self, *args, **kwargs):
        super(BuilderProxy, self).__init__(*args, **kwargs)
        assert isinstance(self, XsdComponent)
        if not cls.is_dummy_component(self):
            cls.components.append(self)
        else:
            cls.dummy_components.append(self)