import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@_registered_volume_type
class SecretVolume(_VolumeType):
    """Represents a secret as a volume."""

    @classmethod
    def release_tracks(cls):
        return [base.ReleaseTrack.ALPHA]

    @classmethod
    def name(cls):
        return 'secret'

    @classmethod
    def help(cls):
        return 'Represents a secret stored in Secret Manager as a volume.'

    @classmethod
    def required_fields(cls):
        return {'secret': 'The name of the secret in Secret Manager. Must be a secret in the same project being deployed or be an alias mapped in the `run.googleapis.com/secrets` annotation.', 'version': 'The version of the secret to make available in the volume.', 'path': 'The relative path within the volume to mount that version.'}

    @classmethod
    def optional_fields(cls):
        return {}

    @classmethod
    def fill_volume(cls, volume, new_vol, messages):
        src = messages.SecretVolumeSource(secretName=volume['secret'])
        item = messages.KeyToPath(path=volume['path'], key=volume['version'])
        src.items.append(item)
        new_vol.secret = src