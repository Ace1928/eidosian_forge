from .roundtrip import YAML
class User1(object):
    yaml_tag = u'!user'

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag, u'{.name}-{.age}'.format(node, node))

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(*node.value.split('-'))