from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.util.apis import arg_marshalling
from googlecloudsdk.command_lib.util.apis import registry
class MethodRef(object):
    """Encapsulates a method specified on the command line with all its flags.

  This makes use of an ArgumentGenerator to generate and parse all the flags
  that correspond to a method. It provides a simple interface to the command so
  the implementor doesn't need to be aware of which flags were added and
  manually extract them. This knows which flags exist and what method fields
  they correspond to.
  """

    def __init__(self, namespace, method, arg_generator):
        """Creates the MethodRef.

    Args:
      namespace: The argparse namespace that holds the parsed args.
      method: APIMethod, The method.
      arg_generator: arg_marshalling.AutoArgumentGenerator, The generator for
        this method.
    """
        self.namespace = namespace
        self.method = method
        self.arg_generator = arg_generator

    def Call(self):
        """Execute the method.

    Returns:
      The result of the method call.
    """
        raw = self.arg_generator.raw
        request = self.arg_generator.CreateRequest(self.namespace)
        limit = self.arg_generator.Limit(self.namespace)
        page_size = self.arg_generator.PageSize(self.namespace)
        return self.method.Call(request, raw=raw, limit=limit, page_size=page_size)