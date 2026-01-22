from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetChunkifiedFilters(self):
    """Returns a list of filter strings where each filter has an upper limit of resource filters.

    The upper limit of resource filters in a contructed filter string is set
    by self._max_resource_chunk_size. This is to avoid having too many
    filters in one API request and getting the request rejected.


    For example, consider this ContainerAnalysisFilter object:
      ContainerAnalysisFilter() \\
        .WithKinds('VULNERABILITY') \\
        .WithResources([
          'url/to/resources/1', 'url/to/resources/2', 'url/to/resources/3',
          'url/to/resources/4', 'url/to/resources/5', 'url/to/resources/6'])

    Calling GetChunkifiedFilters will return the following result:
    [
      '''(kind="VULNERABILITY") AND (resource_url="'url/to/resources/1)"
       OR ("resource_url="'url/to/resources/2")
       OR ("resource_url="'url/to/resources/3")
       OR ("resource_url="'url/to/resources/4")
       OR ("resource_url="'url/to/resources/5")''',
      '(kind="VULNERABILITY") AND (resource_url="url/to/resources/6")'
    ]
    """
    kinds = _OrJoinFilters(*[_HasField('kind', k) for k in self._kinds])
    resources = [_HasField('resourceUrl', r) for r in self._resources]
    base_filter = _AndJoinFilters(_HasPrefixes('resourceUrl', self.resource_prefixes), self.custom_filter, kinds)
    if not resources:
        return [base_filter]
    chunks = [resources[i:i + self._max_resource_chunk_size] for i in range(0, len(resources), self._max_resource_chunk_size)]
    return [_AndJoinFilters(base_filter, _OrJoinFilters(*chunk)) for chunk in chunks]