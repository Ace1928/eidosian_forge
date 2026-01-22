from glance.common import exception
import glance.domain.proxy
class ExtraPropertiesProxy(glance.domain.ExtraProperties):

    def __init__(self, context, extra_props, property_rules):
        self.context = context
        self.property_rules = property_rules
        extra_properties = {}
        for key in extra_props.keys():
            if self.property_rules.check_property_rules(key, 'read', self.context):
                extra_properties[key] = extra_props[key]
        super(ExtraPropertiesProxy, self).__init__(extra_properties)

    def __getitem__(self, key):
        if self.property_rules.check_property_rules(key, 'read', self.context):
            return dict.__getitem__(self, key)
        else:
            raise KeyError

    def __setitem__(self, key, value):
        try:
            if self.__getitem__(key) is not None:
                if self.property_rules.check_property_rules(key, 'update', self.context):
                    return dict.__setitem__(self, key, value)
                else:
                    raise exception.ReservedProperty(property=key)
        except KeyError:
            if self.property_rules.check_property_rules(key, 'create', self.context):
                return dict.__setitem__(self, key, value)
            else:
                raise exception.ReservedProperty(property=key)

    def __delitem__(self, key):
        if key not in super(ExtraPropertiesProxy, self).keys():
            raise KeyError
        if self.property_rules.check_property_rules(key, 'delete', self.context):
            return dict.__delitem__(self, key)
        else:
            raise exception.ReservedProperty(property=key)