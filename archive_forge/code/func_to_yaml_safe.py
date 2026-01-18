from collections.abc import Mapping
def to_yaml_safe(dumper, data):
    """ Converts Munch to a normal mapping node, making it appear as a
            dict in the YAML output.

            >>> b = Munch(foo=['bar', Munch(lol=True)], hello=42)
            >>> import yaml
            >>> yaml.safe_dump(b, default_flow_style=True)
            '{foo: [bar, {lol: true}], hello: 42}\\n'
        """
    return dumper.represent_dict(data)