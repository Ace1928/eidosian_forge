from .roundtrip import YAML
def test_register_1_safe(self):
    yaml = YAML(typ='safe')
    yaml.register_class(User1)
    ys = '\n        [!user Anthon-18]\n        '
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)