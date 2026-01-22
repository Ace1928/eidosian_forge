from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.util.apis import arg_marshalling
from googlecloudsdk.command_lib.util.apis import registry
class MethodDynamicPositionalAction(parser_extensions.DynamicPositionalAction):
    """A DynamicPositionalAction that adds flags for a given method to the parser.

  Based on the value given for method, it looks up the valid fields for that
  method call and adds those flags to the parser.
  """

    def __init__(self, *args, **kwargs):
        self._dest = kwargs.pop('dest')
        super(MethodDynamicPositionalAction, self).__init__(*args, **kwargs)

    def GenerateArgs(self, namespace, method_name):
        full_collection_name = getattr(namespace, 'collection', None)
        api_version = getattr(namespace, 'api_version', None)
        if not full_collection_name:
            raise c_exc.RequiredArgumentException('--collection', 'The collection name must be specified before the API method.')
        method = registry.GetMethod(full_collection_name, method_name, api_version=api_version)
        arg_generator = arg_marshalling.AutoArgumentGenerator(method, raw=namespace.raw)
        method_ref = MethodRef(namespace, method, arg_generator)
        setattr(namespace, self._dest, method_ref)
        return arg_generator.GenerateArgs()

    def Completions(self, prefix, parsed_args, **kwargs):
        return MethodCompleter(prefix, parsed_args, **kwargs)