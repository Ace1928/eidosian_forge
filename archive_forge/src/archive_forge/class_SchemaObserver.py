from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
class SchemaObserver:
    """
    Observer that registers created components. Run the 'clear' method after each usage.
    """
    components = []
    dummy_components = []

    @classmethod
    def observed_builder(cls, builder):
        if isinstance(builder, type):

            class BuilderProxy(builder):

                def __init__(self, *args, **kwargs):
                    super(BuilderProxy, self).__init__(*args, **kwargs)
                    assert isinstance(self, XsdComponent)
                    if not cls.is_dummy_component(self):
                        cls.components.append(self)
                    else:
                        cls.dummy_components.append(self)
            BuilderProxy.__name__ = builder.__name__
            return BuilderProxy
        elif callable(builder):

            @wraps(builder)
            def builder_proxy(*args, **kwargs):
                obj = builder(*args, **kwargs)
                assert isinstance(obj, XsdComponent)
                if not cls.is_dummy_component(obj):
                    cls.components.append(obj)
                else:
                    cls.dummy_components.append(obj)
                return obj
            return builder_proxy

    @classmethod
    def clear(cls) -> None:
        del cls.components[:]
        del cls.dummy_components[:]

    @classmethod
    def is_dummy_component(cls, component) -> bool:
        if component.parent in cls.dummy_components:
            return True
        elif isinstance(component, XsdAttributeGroup):
            return not component
        elif isinstance(component, XsdComplexType):
            return component.name == XSD_ANY_TYPE and component.target_namespace != XSD_NAMESPACE
        elif isinstance(component, XsdGroup) and component.parent is not None:
            return component.parent.name == XSD_ANY_TYPE and component.target_namespace != XSD_NAMESPACE
        return False