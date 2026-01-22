from . import base
from cliff import columns
class YAMLFormatter(base.ListFormatter, base.SingleFormatter):

    def add_argument_group(self, parser):
        pass

    def emit_list(self, column_names, data, stdout, parsed_args):
        import yaml
        items = []
        for item in data:
            items.append({n: _yaml_friendly(i) for n, i in zip(column_names, item)})
        yaml.safe_dump(items, stream=stdout, default_flow_style=False)

    def emit_one(self, column_names, data, stdout, parsed_args):
        import yaml
        for key, value in zip(column_names, data):
            dict_data = {key: _yaml_friendly(value)}
            yaml.safe_dump(dict_data, stream=stdout, default_flow_style=False)