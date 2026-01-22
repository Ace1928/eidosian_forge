from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
class ValidateYAML(base.Command):
    """Validate a YAML file against a JSON Schema.

  {command} validates YAML / JSON files against
  [JSON Schemas](https://json-schema.org/).
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('schema_file', help='The path to a file containing the JSON Schema.')
        parser.add_argument('yaml_file', help='The path to a file containing YAML / JSON data. Use `-` for the standard input.')

    def Run(self, args):
        contents = console_io.ReadFromFileOrStdin(args.yaml_file, binary=False)
        parsed_yaml = yaml.load(contents)
        yaml_validator.Validator(args.schema_file).Validate(parsed_yaml)