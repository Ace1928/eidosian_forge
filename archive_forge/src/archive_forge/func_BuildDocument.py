from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def BuildDocument(self):
    """Build new document.

    The object built by this method becomes the top level entity
    that the builder handler constructs.  The actual type is
    determined by the sub-class of the Builder class and can essentially
    be any type at all.  This method is always called when the parser
    encounters the start of a new document.

    Returns:
      New object instance representing concrete document which is
      returned to user via BuilderHandler.GetResults().
    """