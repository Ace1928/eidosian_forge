from __future__ import absolute_import
import os
def LoadSingleClientDeployInfo(client_deploy_info):
    """Returns a ClientDeployInfoExternal from a deploy_info.yaml file or string.

  Args:
    client_deploy_info: The contents of a client_deploy_info.yaml file or
      string, or an open file object.

  Returns:
    A ClientDeployInfoExternal instance which represents the contents of the
    parsed yaml.

  Raises:
    EmptyYaml: when there are no documents in yaml.
    MultipleClientDeployInfo: when there are multiple documents in yaml.
    yaml_errors.EventError: when an error occurs while parsing the yaml.
  """
    builder = yaml_object.ObjectBuilder(ClientDeployInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(client_deploy_info)
    parsed_yaml = handler.GetResults()
    if not parsed_yaml:
        raise EmptyYaml()
    if len(parsed_yaml) > 1:
        raise MultipleClientDeployInfo()
    return parsed_yaml[0]