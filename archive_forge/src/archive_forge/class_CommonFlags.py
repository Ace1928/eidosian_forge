from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
class CommonFlags(FlagDefs):
    """Flags that are common between the gcloud code dev commands."""

    def __init__(self):
        super(CommonFlags, self).__init__()
        self._group_cache = {}

    def AddLocalPort(self):
        self._AddFlag('--local-port', type=int, help='Local port to which the service connection is forwarded. If this flag is not set, then a random port is chosen.')

    def AddSource(self):
        self._AddFlag('--source', help='The directory containing the source to build. If not specified, the current directory is used.')

    def AddServiceName(self):
        self._AddFlag('--service-name', required=False, help='Name of the service.')

    def AddImage(self):
        self._AddFlag('--image', required=False, help='Name for the built image.')

    def AddMemory(self):
        self._AddFlag('--memory', type=arg_parsers.BinarySize(default_unit='B'), help='Container memory limit. Limit is expressed either as an integer representing the number of bytes or an integer followed by a unit suffix. Valid unit suffixes are "B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB", or "PiB".')

    def AddCpu(self):
        self._AddFlag('--cpu', type=arg_parsers.BoundedFloat(lower_bound=0.0), help='Container CPU limit. Limit is expressed as a number of CPUs. Fractional CPU limits are allowed (e.g. 1.5).')

    def AddCloudsqlInstances(self):
        self._AddFlag('--cloudsql-instances', type=arg_parsers.ArgList(), metavar='CLOUDSQL_INSTANCE', help='Cloud SQL instance connection strings. Must be in the form <project>:<region>:<instance>.')

    def AddReadinessProbe(self):
        self._AddFlag('--readiness-probe', default=False, action='store_true', hidden=True, help='Add a readiness probe to the list of containers that delays deployment stabilization until the application app has bound to $PORT')

    def AddServiceConfigPositionalArg(self, include_app_engine_docs=False):
        """_AddFlag for service_config, which has two possible help strings.

    Args:
      include_app_engine_docs: Add paragraph that says app.yaml is allowed.
    """
        help_text = 'service.yaml filename override. Defaults to the first file matching ```*service.dev.yaml``` then ```*service.yaml```, if any exist. This path is relative to the --source dir.'
        if include_app_engine_docs:
            help_text += '\nAn App Engine config path (typically ```app.yaml```) may also be provided here, and we will build with a Cloud Native Computing Foundation Buildpack builder selected from gcr.io/gae-runtimes/buildpacks, according to the App Engine ```runtime``` specified in app.yaml.'
        self._AddFlag('service_config', metavar='SERVICE_CONFIG', nargs='?', help=help_text)

    def AddAllowSecretManagerFlag(self):
        self._AddFlag('--allow-secret-manager', action=arg_parsers.StoreTrueFalseAction, help='Suppress warnings if secrets need to be pulled from secret manager')

    def AddSecrets(self):
        self._AddFlag('--secrets', metavar='KEY=VALUE', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(key_type=six.text_type, value_type=six.text_type), help='List of key-value pairs to set as secrets.')

    def AddCloud(self):
        self._AddFlag('--cloud', default=False, action='store_true', hidden=True, help='deploy code to Cloud Run')
        self._AddFlag('--region', help='region to deploy the dev service', hidden=True)

    def _GetGroup(self, klass):
        if klass not in self._group_cache:
            group = klass()
            self._group_cache[klass] = group
            self._AddOperation(group)
        return self._group_cache[klass]

    def CredentialsGroup(self):
        return self._GetGroup(CredentialFlags)

    def EnvVarsGroup(self):
        return self._GetGroup(EnvVarFlags)

    def BuildersGroup(self):
        return self._GetGroup(BuilderFlags)

    def AddAlphaAndBetaFlags(self, release_track):
        self._AddBetaFlags()
        if release_track == base.ReleaseTrack.ALPHA:
            self._AddAlphaFlags()
        appyaml_support = release_track == base.ReleaseTrack.ALPHA
        self.AddServiceConfigPositionalArg(include_app_engine_docs=appyaml_support)

    def _AddBetaFlags(self):
        """Set up flags that are for alpha and beta tracks."""
        self.BuildersGroup().AddDockerfile()
        self.AddSource()
        self.AddLocalPort()
        self.CredentialsGroup().AddServiceAccount()
        self.CredentialsGroup().AddApplicationDefaultCredential()
        self.AddReadinessProbe()
        self.AddAllowSecretManagerFlag()
        self.AddSecrets()
        self.BuildersGroup().AddBuilder()

    def _AddAlphaFlags(self):
        """Set up flags that are for alpha track only."""
        self.AddCloudsqlInstances()
        self.AddServiceName()
        self.AddImage()
        self.AddMemory()
        self.AddCpu()
        self.EnvVarsGroup().AddEnvVars()
        self.EnvVarsGroup().AddEnvVarsFile()
        self.AddCloud()