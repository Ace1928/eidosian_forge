from .roundtrip import YAML
class TestRegisterClass(object):

    def test_register_0_rt(self):
        yaml = YAML()
        yaml.register_class(User0)
        ys = '\n        - !User0\n          name: Anthon\n          age: 18\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys, unordered_lines=True)

    def test_register_0_safe(self):
        yaml = YAML(typ='safe')
        yaml.register_class(User0)
        ys = '\n        - !User0 {age: 18, name: Anthon}\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_register_0_unsafe(self):
        yaml = YAML(typ='unsafe')
        yaml.register_class(User0)
        ys = '\n        - !User0 {age: 18, name: Anthon}\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_register_1_rt(self):
        yaml = YAML()
        yaml.register_class(User1)
        ys = '\n        - !user Anthon-18\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_register_1_safe(self):
        yaml = YAML(typ='safe')
        yaml.register_class(User1)
        ys = '\n        [!user Anthon-18]\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)

    def test_register_1_unsafe(self):
        yaml = YAML(typ='unsafe')
        yaml.register_class(User1)
        ys = '\n        [!user Anthon-18]\n        '
        d = yaml.load(ys)
        yaml.dump(d, compare=ys)