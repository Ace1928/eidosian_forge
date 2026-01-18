from .roundtrip import YAML
def test_register_1_unsafe(self):
    yaml = YAML(typ='unsafe')
    yaml.register_class(User1)
    ys = '\n        [!user Anthon-18]\n        '
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)