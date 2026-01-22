from glance.common import exception
import glance.domain.proxy
class ProtectedImageProxy(glance.domain.proxy.Image):

    def __init__(self, image, context, property_rules):
        self.image = image
        self.context = context
        self.property_rules = property_rules
        self.image.extra_properties = ExtraPropertiesProxy(self.context, self.image.extra_properties, self.property_rules)
        super(ProtectedImageProxy, self).__init__(self.image)