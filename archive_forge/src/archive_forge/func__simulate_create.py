def _simulate_create(self, attributes, timeout, wait, **kwargs):

    class Resource(dict):

        def to_dict(self, *args, **kwargs):
            return self
    return Resource(attributes)