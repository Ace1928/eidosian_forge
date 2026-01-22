class MetadefPropertyRepo(object):

    def __init__(self, base, property_proxy_class=None, property_proxy_kwargs=None):
        self.base = base
        self.property_proxy_helper = Helper(property_proxy_class, property_proxy_kwargs)

    def get(self, namespace, property_name):
        property = self.base.get(namespace, property_name)
        return self.property_proxy_helper.proxy(property)

    def add(self, property):
        self.base.add(self.property_proxy_helper.unproxy(property))

    def list(self, *args, **kwargs):
        properties = self.base.list(*args, **kwargs)
        return [self.property_proxy_helper.proxy(property) for property in properties]

    def remove(self, item):
        base_item = self.property_proxy_helper.unproxy(item)
        result = self.base.remove(base_item)
        return self.property_proxy_helper.proxy(result)

    def save(self, item):
        base_item = self.property_proxy_helper.unproxy(item)
        result = self.base.save(base_item)
        return self.property_proxy_helper.proxy(result)