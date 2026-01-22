from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
class ExportSchemasGenerator(object):
    """Recursively generates export JSON schemas for nested messages."""

    def __init__(self, api, directory=None):
        self._api = api
        self._directory = directory
        self._generated = set()

    def _GetSchemaFileName(self, message_name):
        """Returns the schema file name given the message name."""
        return message_name + '.yaml'

    def _GetSchemaFilePath(self, message_name):
        """Returns the schema file path name given the message name."""
        file_path = self._GetSchemaFileName(message_name)
        if self._directory:
            file_path = os.path.join(self._directory, file_path)
        return file_path

    def _WriteSchema(self, message_name, spec):
        """Writes the schema in spec to the _GetSchemaFilePath() file."""
        tmp = io.StringIO()
        tmp.write('$schema: "http://json-schema.org/draft-06/schema#"\n\n')
        yaml_printer.YamlPrinter(name='yaml', projector=resource_projector.IdentityProjector(), out=tmp).Print(spec)
        content = re.sub('\n *{}\n'.format(_YAML_WORKAROUND), '\n', tmp.getvalue())
        file_path = self._GetSchemaFilePath(message_name)
        log.info('Generating JSON schema [{}].'.format(file_path))
        with files.FileWriter(file_path) as w:
            w.write(content)

    def _AddFields(self, depth, parent, spec, fields):
        """Adds message fields to the YAML spec.

    Args:
      depth: The nested dict depth.
      parent: The parent spec (nested ordered dict to add fields to) of spec.
      spec: The nested ordered dict to add fields to.
      fields: A message spec fields dict to add to spec.
    """
        depth += 2
        for name, value in sorted(six.iteritems(fields)):
            description = value['description'].strip()
            if description.startswith(_OPTIONAL):
                description = description[len(_OPTIONAL):].strip()
            elif description.startswith(_REQUIRED):
                description = description[len(_REQUIRED):].strip()
            if description.startswith(_OUTPUT_ONLY):
                continue
            d = collections.OrderedDict()
            spec[name] = d
            d['description'] = _WrapDescription(depth, description)
            if value.get('repeated'):
                d['type'] = 'array'
                items = collections.OrderedDict(value.get('items', {}))
                d['items'] = items
                d = items
                depth += 2
            type_name = value.get('type', 'boolean')
            subfields = value.get('fields')
            if subfields:
                if name == 'additionalProperties':
                    del spec[name]
                    properties = collections.OrderedDict()
                    self._AddFields(depth, d, properties, subfields)
                    if properties:
                        parent[name] = properties
                else:
                    d['$ref'] = self._GetSchemaFileName(type_name)
                    self.Generate(type_name, subfields)
            elif type_name in self._generated:
                d['$ref'] = self._GetSchemaFileName(type_name)
            else:
                type_name = _NormalizeTypeName(type_name)
                if type_name == 'enum':
                    enum = value.get('choices')
                    d['type'] = 'string'
                    d['enum'] = sorted([n for n, _ in six.iteritems(enum)])
                else:
                    d['type'] = type_name

    def Generate(self, message_name, message_spec):
        """Recursively generates export/import YAML schemas for message_spec.

    The message and nested messages are generated in separate schema files in
    the destination directory. Pre-existing files are silently overwritten.

    Args:
      message_name: The API message name for message_spec.
      message_spec: An arg_utils.GetRecursiveMessageSpec() message spec.
    """
        if message_name in self._generated:
            return
        self._generated.add(message_name)
        spec = collections.OrderedDict()
        spec['title'] = '{} {} {} export schema'.format(self._api.name, self._api.version, message_name)
        spec['description'] = _SPEC_DESCRIPTION
        spec['type'] = 'object'
        _AddRequiredFields(spec, message_spec)
        spec['additionalProperties'] = False
        properties = collections.OrderedDict()
        spec['properties'] = properties
        type_string = {'type': 'string'}
        comment = collections.OrderedDict()
        properties['COMMENT'] = comment
        comment['type'] = 'object'
        comment['description'] = 'User specified info ignored by gcloud import.'
        comment['additionalProperties'] = False
        comment_properties = collections.OrderedDict()
        comment['properties'] = comment_properties
        comment_properties['template-id'] = collections.OrderedDict(type_string)
        comment_properties['region'] = collections.OrderedDict(type_string)
        comment_properties['description'] = collections.OrderedDict(type_string)
        comment_properties['date'] = collections.OrderedDict(type_string)
        comment_properties['version'] = collections.OrderedDict(type_string)
        unknown = collections.OrderedDict()
        properties['UNKNOWN'] = unknown
        unknown['type'] = 'array'
        unknown['description'] = 'Unknown API fields that cannot be imported.'
        unknown['items'] = type_string
        self._AddFields(1, spec, properties, message_spec)
        self._WriteSchema(message_name, spec)