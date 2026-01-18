from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
Create a Version object.

    The object is based on an optional YAML configuration file and the
    parameters to this method; any provided method parameters override any
    provided in-file configuration.

    The file may only have the fields given in
    VersionsClientBase._ALLOWED_YAML_FIELDS specified; the only parameters
    allowed are those that can be specified on the command line.

    Args:
      name: str, the name of the version object to create.
      path: str, the path to the YAML file.
      deployment_uri: str, the deploymentUri to set for the Version
      runtime_version: str, the runtimeVersion to set for the Version
      labels: Version.LabelsValue, the labels to set for the version
      machine_type: str, the machine type to serve the model version on.
      description: str, the version description.
      framework: FrameworkValueValuesEnum, the ML framework used to train this
        version of the model.
      python_version: str, The version of Python used to train the model.
      prediction_class: str, the FQN of a Python class implementing the Model
        interface for custom prediction.
      package_uris: list of str, Cloud Storage URIs containing user-supplied
        Python code to use.
      accelerator_config: an accelerator config message object.
      service_account: Specifies the service account for resource access
        control.
      explanation_method: Enables explanations and selects the explanation
        method. Valid options are 'integrated-gradients' and 'sampled-shapley'.
      num_integral_steps: Number of integral steps for Integrated Gradients and
        XRAI.
      num_paths: Number of paths for Sampled Shapley.
      image: The container image to deploy.
      command: Entrypoint for the container image.
      container_args: The command-line args to pass the container.
      env_vars: The environment variables to set on the container.
      ports: The ports to which traffic will be sent in the container.
      predict_route: The HTTP path within the container that predict requests
        are sent to.
      health_route: The HTTP path within the container that health checks are
        sent to.
      min_nodes: The minimum number of nodes to scale this model under load.
      max_nodes: The maximum number of nodes to scale this model under load.
      metrics: List of key-value pairs to set as metrics' target for
        autoscaling.
      containers_hidden: Whether or not container-related fields are hidden on
        this track.
      autoscaling_hidden: Whether or not autoscaling fields are hidden on this
        track.

    Returns:
      A Version object (for the corresponding API version).

    Raises:
      InvalidVersionConfigFile: If the file contains unexpected fields.
    