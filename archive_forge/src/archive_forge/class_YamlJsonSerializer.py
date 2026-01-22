import json
import os
import betamax.serializers.base
import yaml
class YamlJsonSerializer(betamax.serializers.base.BaseSerializer):
    name = 'yamljson'

    @staticmethod
    def generate_cassette_name(cassette_library_dir, cassette_name):
        return os.path.join(cassette_library_dir, '{name}.yaml'.format(name=cassette_name))

    def serialize(self, cassette_data):
        for interaction in cassette_data['http_interactions']:
            for key in ('request', 'response'):
                if _is_json_body(interaction[key]):
                    interaction[key]['body']['string'] = _indent_json(interaction[key]['body']['string'])

        class MyDumper(yaml.Dumper):
            """Specialized Dumper which does nice blocks and unicode."""
        yaml.representer.BaseRepresenter.represent_scalar = _represent_scalar
        MyDumper.add_representer(str, _unicode_representer)
        return yaml.dump(cassette_data, Dumper=MyDumper, default_flow_style=False)

    def deserialize(self, cassette_data):
        try:
            deserialized = yaml.safe_load(cassette_data)
        except yaml.error.YAMLError:
            deserialized = None
        if deserialized is not None:
            return deserialized
        return {}