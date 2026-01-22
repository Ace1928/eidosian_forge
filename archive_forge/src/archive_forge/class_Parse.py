from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
import six
from six.moves import range
class Parse(base.Command):
    """Cloud SDK resource test URI generator.

  *{command}* is an handy way to generate test URIs for the resource parser.
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--collection', metavar='NAME', required=True, help='The resource collection name of the resource to parse.')
        parser.add_argument('--api-version', metavar='VERSION', help='The resource collection API version. The collection default is used if not specified.')
        parser.add_argument('--count', default=1, type=arg_parsers.BoundedInt(lower_bound=1), help='The number of test resource URIs to generate.')

    def Run(self, args):
        """Returns the list of generated resources."""
        collection_info = resources.REGISTRY.GetCollectionInfo(args.collection, api_version=args.api_version)
        templates = {}
        params = collection_info.GetParams('')
        if not params:
            return []
        for param in params:
            templates[param] = 'my-' + param.lower() + '-{}'
        uris = []
        for i in range(1, args.count + 1):
            params = {}
            for param, template in six.iteritems(templates):
                params[param] = template.format(i)
            uri = resources.Resource(None, collection_info, '', params, None).SelfLink()
            uris.append(uri)
        return uris