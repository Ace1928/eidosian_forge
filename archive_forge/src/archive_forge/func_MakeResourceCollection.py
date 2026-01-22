from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
def MakeResourceCollection(self, collection_name, path, enable_uri_parsing, api_version):
    _, url_api_version, _ = resource_util.SplitEndpointUrl(self.base_url)
    if url_api_version:
        base_url = self.base_url
    else:
        base_url = '{}{}/'.format(self.base_url, api_version)
    return resource_util.CollectionInfo(self.api_name, api_version, base_url, self.docs_url, collection_name, path, {}, resource_util.GetParamsFromPath(path), enable_uri_parsing)