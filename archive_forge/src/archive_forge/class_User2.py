from .roundtrip import YAML
@yaml_object(yml)
class User2(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age