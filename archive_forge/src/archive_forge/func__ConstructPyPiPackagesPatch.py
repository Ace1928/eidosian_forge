from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructPyPiPackagesPatch(clear_pypi_packages, remove_pypi_packages, update_pypi_packages, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for partially updating PyPI packages.

  Args:
    clear_pypi_packages: bool, whether to clear the PyPI packages dictionary.
    remove_pypi_packages: iterable(string), Iterable of PyPI package names to
      remove.
    update_pypi_packages: {string: string}, dict mapping PyPI package name to
      optional extras and version specifier.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    env_cls = messages.Environment
    pypi_packages_cls = messages.SoftwareConfig.PypiPackagesValue
    entry_cls = pypi_packages_cls.AdditionalProperty

    def _BuildEnv(entries):
        software_config = messages.SoftwareConfig(pypiPackages=pypi_packages_cls(additionalProperties=entries))
        config = messages.EnvironmentConfig(softwareConfig=software_config)
        return env_cls(config=config)
    return command_util.BuildPartialUpdate(clear_pypi_packages, remove_pypi_packages, update_pypi_packages, 'config.software_config.pypi_packages', entry_cls, _BuildEnv)