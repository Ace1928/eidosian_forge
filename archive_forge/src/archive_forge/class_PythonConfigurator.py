from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
class PythonConfigurator(ext_runtime.Configurator):
    """Generates configuration for a Python application."""

    def __init__(self, path, params, runtime):
        """Constructor.

    Args:
      path: (str) Root path of the source tree.
      params: (ext_runtime.Params) Parameters passed through to the
        fingerprinters.
      runtime: (str) The runtime name.
    """
        self.root = path
        self.params = params
        self.runtime = runtime

    def GenerateAppYaml(self, notify):
        """Generate app.yaml.

    Args:
      notify: depending on whether we're in deploy, write messages to the
        user or to log.
    Returns:
      (bool) True if file was written

    Note: this is not a recommended use-case,
    python-compat users likely have an existing app.yaml.  But users can
    still get here with the --runtime flag.
    """
        if not self.params.appinfo:
            app_yaml = os.path.join(self.root, 'app.yaml')
            if not os.path.exists(app_yaml):
                notify('Writing [app.yaml] to [%s].' % self.root)
                runtime = 'custom' if self.params.custom else self.runtime
                files.WriteFileContents(app_yaml, PYTHON_APP_YAML.format(runtime=runtime))
                log.warning(APP_YAML_WARNING)
                return True
        return False

    def GenerateDockerfileData(self):
        """Generates dockerfiles.

    Returns:
      list(ext_runtime.GeneratedFile) the list of generated dockerfiles
    """
        if self.runtime == 'python-compat':
            dockerfile_preamble = COMPAT_DOCKERFILE_PREAMBLE
        else:
            dockerfile_preamble = PYTHON27_DOCKERFILE_PREAMBLE
        all_config_files = []
        dockerfile_name = config.DOCKERFILE
        dockerfile_components = [dockerfile_preamble, DOCKERFILE_INSTALL_APP]
        if self.runtime == 'python-compat':
            dockerfile_components.append(DOCKERFILE_INSTALL_REQUIREMENTS_TXT)
        dockerfile_contents = ''.join((c for c in dockerfile_components))
        dockerfile = ext_runtime.GeneratedFile(dockerfile_name, dockerfile_contents)
        all_config_files.append(dockerfile)
        dockerignore = ext_runtime.GeneratedFile('.dockerignore', DOCKERIGNORE)
        all_config_files.append(dockerignore)
        return all_config_files

    def GenerateConfigs(self):
        """Generate all config files for the module."""
        notify = log.info if self.params.deploy else log.status.Print
        self.GenerateAppYaml(notify)
        created = False
        if self.params.custom or self.params.deploy:
            dockerfiles = self.GenerateDockerfileData()
            for dockerfile in dockerfiles:
                if dockerfile.WriteTo(self.root, notify):
                    created = True
            if not created:
                notify('All config files already exist, not generating anything.')
        return created

    def GenerateConfigData(self):
        """Generate all config files for the module.

    Returns:
      list(ext_runtime.GeneratedFile) A list of the config files
        that were generated
    """
        notify = log.info if self.params.deploy else log.status.Print
        self.GenerateAppYaml(notify)
        if not (self.params.custom or self.params.deploy):
            return []
        all_config_files = self.GenerateDockerfileData()
        return [f for f in all_config_files if not os.path.exists(os.path.join(self.root, f.filename))]