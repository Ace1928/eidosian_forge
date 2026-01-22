from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def AppengineWebMatcher(path, stager, appyaml):
    """Generate a Service from an appengine-web.xml source path.

  This function is a path matcher that returns if and only if:
  - `path` points to either `.../WEB-INF/appengine-web.xml` or `<app-dir>` where
    `<app-dir>/WEB-INF/appengine-web.xml` exists.
  - the xml-file is a valid appengine-web.xml file according to the Java stager.

  The service will be staged according to the stager as a java-xml runtime,
  which is defined in staging.py.

  Args:
    path: str, Unsanitized absolute path, may point to a directory or a file of
        any type. There is no guarantee that it exists.
    stager: staging.Stager, stager that will be invoked if there is a runtime
        and environment match.
    appyaml: str or None, the app.yaml location to used for deployment.

  Raises:
    staging.StagingCommandFailedError, staging command failed.

  Returns:
    Service, fully populated with entries that respect a staged deployable
        service, or None if the path does not match the pattern described.
  """
    suffix = os.path.join(os.sep, 'WEB-INF', 'appengine-web.xml')
    app_dir = path[:-len(suffix)] if path.endswith(suffix) else path
    descriptor = os.path.join(app_dir, 'WEB-INF', 'appengine-web.xml')
    if not os.path.isfile(descriptor):
        return None
    xml_file = files.ReadFileContents(descriptor)
    if '<application>' in xml_file or '<version>' in xml_file:
        log.warning('<application> and <version> elements in ' + '`appengine-web.xml` are not respected')
    staging_dir = stager.Stage(descriptor, app_dir, 'java-xml', env.STANDARD, appyaml)
    if not staging_dir:
        return None
    yaml_path = os.path.join(staging_dir, 'app.yaml')
    service_info = yaml_parsing.ServiceYamlInfo.FromFile(yaml_path)
    return Service(descriptor, app_dir, service_info, staging_dir)