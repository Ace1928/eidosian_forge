import math
class VectorDescriptorNamespace(dict):

    def __init__(self, **kwargs):
        self.update(kwargs)