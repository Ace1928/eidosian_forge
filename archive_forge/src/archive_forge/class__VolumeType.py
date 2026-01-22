import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
class _VolumeType(abc.ABC):
    """Base class for supported volume types.

  To add a new supported volume type, create a subclass of this type,
  implement all the abstract methods, and annotate the class with
  @_registered_volume_type.
  """

    @classmethod
    @abc.abstractmethod
    def name(cls):
        """The name of this Volume type.

    This is the string that will need to be provided as the `type` value in the
    add volumes flag to use this type of volume.
    """
        pass

    @classmethod
    @abc.abstractmethod
    def help(cls):
        """Help text for this volume type."""
        pass

    @classmethod
    def release_tracks(cls):
        """The list of release tracks that this volume type should be present in.

    Used to progressively roll out types of volumes.

    Returns:
      A list of base.ReleaseTrack
    """
        return base.ReleaseTrack.AllValues()

    @classmethod
    @abc.abstractmethod
    def required_fields(cls):
        """A dict of field_name: help text for all fields that must be present."""
        pass

    @classmethod
    @abc.abstractmethod
    def optional_fields(cls):
        """A dict of field_name: help text for all fields that are optional."""
        pass

    @classmethod
    @abc.abstractmethod
    def fill_volume(cls, volume, new_vol, messages):
        """Fills in the Volume message from the provided volume dict."""
        pass

    @classmethod
    def validate_volume_add(cls, volume):
        """Validate that the volume dict has all needed parameters for this type."""
        required_keys = set(cls.required_fields().keys())
        optional_keys = set(cls.optional_fields().keys())
        for key in volume:
            if key == 'name':
                continue
            elif key == 'type':
                if volume[key] != cls.name():
                    raise serverless_exceptions.ConfigurationError('expected volume of type {} but got {}'.format(cls.name(), volume[key]))
            elif key not in required_keys and key not in optional_keys:
                raise serverless_exceptions.ConfigurationError('Volume {} of type {} had unexpected parameter {}'.format(volume['name'], volume['type'], key))
        missing = required_keys - volume.keys()
        if missing:
            raise serverless_exceptions.ConfigurationError('Volume {} of type {} requires the following parameters: {}'.format(volume['name'], volume['type'], missing))

    @classmethod
    def generate_help(cls):
        """Generate help text for this volume type."""
        required_fields = '\n'.join(('* {}: (required) {}  '.format(name, hlp) for name, hlp in cls.required_fields().items()))
        required = f'\n{required_fields}  ' if required_fields.strip() else ''
        optional_fields = '\n'.join(('* {}: (optional) {}  '.format(name, hlp) for name, hlp in cls.optional_fields().items()))
        optional = f'\n{optional_fields}  ' if optional_fields.strip() else ''
        return '*{name}*: {hlp}\n  Additional keys:  {required}{optional}  '.format(name=cls.name(), hlp=cls.help(), required=required, optional=optional)