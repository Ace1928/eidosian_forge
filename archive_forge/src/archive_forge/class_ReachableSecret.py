from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
class ReachableSecret(object):
    """A named secret+version that we can use in an env var or volume mount.

  See CL notes for references to the syntax being parsed here.

  This same type is used for local secrets in this project and remote secrets
  that are mapped in the run.googleapis.com/secrets annotation. This class
  adds to that annotation as needed.
  """
    _PROJECT_NUMBER_PARTIAL = '(?P<project>[0-9]{1,19})'
    _SECRET_NAME_PARTIAL = '(?P<secret>[a-zA-Z0-9-_]{1,255})'
    _REMOTE_SECRET_VERSION_SHORT = ':(?P<version_short>.+)'
    _REMOTE_SECRET_VERSION_LONG = '/versions/(?P<version_long>.+)'
    _REMOTE_SECRET_VERSION = '(?:' + _REMOTE_SECRET_VERSION_SHORT + '|' + _REMOTE_SECRET_VERSION_LONG + ')?'
    _REMOTE_SECRET_FLAG_VALUE = '^projects/' + _PROJECT_NUMBER_PARTIAL + '/secrets/' + _SECRET_NAME_PARTIAL + _REMOTE_SECRET_VERSION + '$'

    @staticmethod
    def IsRemotePath(secret_name):
        return bool(re.search(ReachableSecret._REMOTE_SECRET_FLAG_VALUE, secret_name))

    def __init__(self, flag_value, connector_name, force_managed=False):
        """Parse flag value to make a ReachableSecret.

    Args:
      flag_value: str. A secret identifier like 'sec1:latest'. See tests for
        other supported formats (which vary by platform).
      connector_name: Union[str, PATH_OR_ENV].  An env var ('ENV1') or a mount
        point ('/a/b') for use in error messages. Also used in validation since
        you can only use MOUNT_ALL mode with a mount path.
      force_managed: bool. If True, always use the behavior of managed Cloud Run
        even if the platform is set to gke. Used by Cloud Run local development.
    """
        self._connector = connector_name
        self.force_managed = force_managed
        if force_managed or platforms.IsManaged():
            match = re.search(self._REMOTE_SECRET_FLAG_VALUE, flag_value)
            if match:
                self.remote_project_number = match.group('project')
                self.secret_name = match.group('secret')
                self.secret_version = match.group('version_short')
                if self.secret_version is None:
                    self.secret_version = match.group('version_long')
                if self.secret_version is None:
                    self.secret_version = 'latest'
                return
        self._InitWithLocalSecret(flag_value, connector_name)

    def _InitWithLocalSecret(self, flag_value, connector_name):
        """Init this ReachableSecret for a simple, non-remote secret.

    Args:
      flag_value: str. A secret identifier like 'sec1:latest'. See tests for
        other supported formats.
      connector_name: Union[str, PATH_OR_ENV]. An env var, a mount point, or
        PATH_OR_ENV. See __init__ docs.

    Raises:
      ValueError on flag value parse failure.
    """
        self.remote_project_number = None
        parts = flag_value.split(':')
        if len(parts) == 1:
            self.secret_name, = parts
            self.secret_version = self._OmittedSecretKeyDefault(connector_name)
        elif len(parts) == 2:
            self.secret_name, self.secret_version = parts
        else:
            raise ValueError('Invalid secret spec %r' % flag_value)
        self._AssertValidLocalSecretName(self.secret_name)

    def __repr__(self):
        version_display = self.secret_version
        if self.secret_version == SpecialVersion.MOUNT_ALL:
            version_display = version_display.name
        project_display = 'project=%s ' % self.remote_project_number if self.remote_project_number is not None else ''
        return '<ReachableSecret {project_display}name={secret_name} version={version_display}>'.format(project_display=project_display, secret_name=self.secret_name, version_display=version_display)

    def __eq__(self, other):
        return self.remote_project_number == other.remote_project_number and self.secret_name == other.secret_name and (self.secret_version == other.secret_version)

    def __ne__(self, other):
        return not self == other

    def _OmittedSecretKeyDefault(self, name):
        """The key/version value to use for a secret flag that has no version.

    Args:
      name: The env var or mount point, for use in an error message.

    Returns:
      str value to use as the secret version.

    Raises:
      ConfigurationError: If the key is required on this platform.
    """
        if self.force_managed or platforms.IsManaged():
            raise exceptions.ConfigurationError('No secret version specified for {name}. Use {name}:latest to reference the latest version.'.format(name=name))
        else:
            if self._connector is SpecialConnector.PATH_OR_ENV:
                raise TypeError("Can't determine default key for secret named %r." % name)
            if not self._connector.startswith('/'):
                raise exceptions.ConfigurationError('Missing required item key for the secret at [{}].'.format(name))
            else:
                return SpecialVersion.MOUNT_ALL

    def _AssertValidLocalSecretName(self, name):
        if not re.search('^' + self._SECRET_NAME_PARTIAL + '$', name):
            raise exceptions.ConfigurationError('%r is not a valid secret name.' % name)

    def _PathTail(self):
        """Last path component of self._connector."""
        if self._connector is SpecialConnector.PATH_OR_ENV:
            raise TypeError("Can't make SecretVolumeSource message for secret %r of unknown usage." % self.secret_name)
        if not self._connector.startswith('/'):
            raise TypeError("Can't make SecretVolumeSource message for secret connected to env var %r." % self._connector)
        return self._connector.rsplit('/', 1)[-1]

    def _IsRemote(self):
        return self.remote_project_number is not None

    def _GetOrCreateAlias(self, resource):
        """What do we call this secret within this resource?

    Note that there might be an existing alias to the same secret, which we'd
    like to reuse. There's no effort to deduplicate the ReachableSecret python
    objects; you just get the same alias from more than one of them.

    The k8s_object annotation is edited here to include all new aliases. Use
    PruneAnnotation to clean up unused ones.

    Args:
      resource: k8s_object resource that will be modified if we need to add a
        new alias to the secrets annotation.

    Returns:
      str for use as SecretVolumeSource.secret_name or SecretKeySelector.name
    """
        if not self._IsRemote():
            return self.secret_name
        formatted_annotation = _GetSecretsAnnotation(resource)
        remotes = ParseAnnotation(formatted_annotation)
        for alias, other_rs in remotes.items():
            if self == other_rs:
                return alias
        new_alias = self.secret_name[:5] + '-' + str(uuid.uuid1())
        remotes[new_alias] = self
        _SetSecretsAnnotation(resource, _FormatAnnotation(remotes))
        return new_alias

    def FormatAnnotationItem(self):
        """Render a secret path for the run.googleapis.com/secrets annotation.

    Returns:
      str like 'projects/123/secrets/s1'

    Raises:
      TypeError for a local secret name that doesn't belong in the annotation.
    """
        if not self._IsRemote():
            raise TypeError('Only remote paths go in annotations')
        return 'projects/{remote_project_number}/secrets/{secret_name}'.format(remote_project_number=self.remote_project_number, secret_name=self.secret_name)

    def AsSecretVolumeSource(self, resource):
        """Build message for adding to resource.template.volumes.secrets.

    Args:
      resource: k8s_object that may get modified with new aliases.

    Returns:
      messages.SecretVolumeSource
    """
        if platforms.IsManaged():
            return self._AsSecretVolumeSource_ManagedMode(resource)
        else:
            return self._AsSecretVolumeSource_NonManagedMode(resource)

    def AppendToSecretVolumeSource(self, resource, out):
        messages = resource.MessagesModule()
        item = messages.KeyToPath(path=self._PathTail(), key=self.secret_version)
        out.items.append(item)

    def _AsSecretVolumeSource_ManagedMode(self, resource):
        messages = resource.MessagesModule()
        out = messages.SecretVolumeSource(secretName=self._GetOrCreateAlias(resource))
        self.AppendToSecretVolumeSource(resource, out)
        return out

    def _AsSecretVolumeSource_NonManagedMode(self, resource):
        messages = resource.MessagesModule()
        out = messages.SecretVolumeSource(secretName=self._GetOrCreateAlias(resource))
        if self.secret_version != SpecialVersion.MOUNT_ALL:
            out.items.append(messages.KeyToPath(key=self.secret_version, path=self.secret_version))
        return out

    def AsEnvVarSource(self, resource):
        """Build message for adding to resource.template.env_vars.secrets.

    Args:
      resource: k8s_object that may get modified with new aliases.

    Returns:
      messages.EnvVarSource
    """
        messages = resource.MessagesModule()
        selector = messages.SecretKeySelector(name=self._GetOrCreateAlias(resource), key=self.secret_version)
        return messages.EnvVarSource(secretKeyRef=selector)