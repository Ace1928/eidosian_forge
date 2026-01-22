from glance.common import exception
import glance.domain.proxy
class ProtectedImageFactoryProxy(glance.domain.proxy.ImageFactory):

    def __init__(self, image_factory, context, property_rules):
        self.image_factory = image_factory
        self.context = context
        self.property_rules = property_rules
        kwargs = {'context': self.context, 'property_rules': self.property_rules}
        super(ProtectedImageFactoryProxy, self).__init__(image_factory, proxy_class=ProtectedImageProxy, proxy_kwargs=kwargs)

    def new_image(self, **kwargs):
        extra_props = kwargs.pop('extra_properties', {})
        extra_properties = {}
        for key in extra_props.keys():
            if self.property_rules.check_property_rules(key, 'create', self.context):
                extra_properties[key] = extra_props[key]
            else:
                raise exception.ReservedProperty(property=key)
        return super(ProtectedImageFactoryProxy, self).new_image(extra_properties=extra_properties, **kwargs)