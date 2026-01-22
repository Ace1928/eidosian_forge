from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def AddOptions(messages, options_file, type_provider):
    """Parse api options from the file and add them to type_provider.

  Args:
    messages: The API message to use.
    options_file: String path expression pointing to a type-provider options
      file.
    type_provider: A TypeProvider message on which the options will be set.

  Returns:
    The type_provider after applying changes.
  Raises:
    exceptions.ConfigError: the api options file couldn't be parsed as yaml
  """
    if not options_file:
        return type_provider
    yaml_content = yaml.load_path(options_file)
    if yaml_content:
        if 'collectionOverrides' in yaml_content:
            type_provider.collectionOverrides = []
            for collection_override_data in yaml_content['collectionOverrides']:
                collection_override = messages.CollectionOverride(collection=collection_override_data['collection'])
                if 'options' in collection_override_data:
                    collection_override.options = _OptionsFrom(messages, collection_override_data['options'])
                type_provider.collectionOverrides.append(collection_override)
        if 'options' in yaml_content:
            type_provider.options = _OptionsFrom(messages, yaml_content['options'])
        if 'credential' in yaml_content:
            type_provider.credential = _CredentialFrom(messages, yaml_content['credential'])
    return type_provider