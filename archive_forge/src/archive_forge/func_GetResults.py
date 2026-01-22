from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def GetResults(self):
    """Get results of document stream processing.

    This method can be invoked after fully parsing the entire YAML file
    to retrieve constructed contents of YAML file.  Called after EndStream.

    Returns:
      A tuple of all document objects that were parsed from YAML stream.

    Raises:
      InternalError if the builder stack is not empty by the end of parsing.
    """
    if self._stack is not None:
        raise yaml_errors.InternalError('Builder stack is not empty.')
    return tuple(self._results)