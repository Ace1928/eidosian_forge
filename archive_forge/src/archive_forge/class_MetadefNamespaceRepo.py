class MetadefNamespaceRepo(object):

    def __init__(self, base, namespace_proxy_class=None, namespace_proxy_kwargs=None):
        self.base = base
        self.namespace_proxy_helper = Helper(namespace_proxy_class, namespace_proxy_kwargs)

    def get(self, namespace):
        namespace_obj = self.base.get(namespace)
        return self.namespace_proxy_helper.proxy(namespace_obj)

    def add(self, namespace):
        self.base.add(self.namespace_proxy_helper.unproxy(namespace))

    def list(self, *args, **kwargs):
        namespaces = self.base.list(*args, **kwargs)
        return [self.namespace_proxy_helper.proxy(namespace) for namespace in namespaces]

    def remove(self, item):
        base_item = self.namespace_proxy_helper.unproxy(item)
        result = self.base.remove(base_item)
        return self.namespace_proxy_helper.proxy(result)

    def remove_objects(self, item):
        base_item = self.namespace_proxy_helper.unproxy(item)
        result = self.base.remove_objects(base_item)
        return self.namespace_proxy_helper.proxy(result)

    def remove_properties(self, item):
        base_item = self.namespace_proxy_helper.unproxy(item)
        result = self.base.remove_properties(base_item)
        return self.namespace_proxy_helper.proxy(result)

    def remove_tags(self, item):
        base_item = self.namespace_proxy_helper.unproxy(item)
        result = self.base.remove_tags(base_item)
        return self.namespace_proxy_helper.proxy(result)

    def save(self, item):
        base_item = self.namespace_proxy_helper.unproxy(item)
        result = self.base.save(base_item)
        return self.namespace_proxy_helper.proxy(result)