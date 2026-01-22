from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class ValidatedBase(object):
    """Base class for all validated objects."""

    @classmethod
    def GetValidator(cls, key):
        """Safely get the Validator corresponding to the given key.

    This function should be overridden by subclasses

    Args:
      key: The attribute or item to get a validator for.

    Returns:
      Validator associated with key or attribute.

    Raises:
      ValidationError: if the requested key is illegal.
    """
        raise NotImplementedError('Subclasses of ValidatedBase must override GetValidator.')

    def SetMultiple(self, attributes):
        """Set multiple values on Validated instance.

    All attributes will be validated before being set.

    Args:
      attributes: A dict of attributes/items to set.

    Raises:
      ValidationError: when no validated attribute exists on class.
    """
        for key, value in attributes.items():
            self.Set(key, value)

    def Set(self, key, value):
        """Set a single value on Validated instance.

    This method should be overridded by sub-classes.

    This method can only be used to assign validated attributes/items.

    Args:
      key: The name of the attributes
      value: The value to set

    Raises:
      ValidationError: when no validated attribute exists on class.
    """
        raise NotImplementedError('Subclasses of ValidatedBase must override Set.')

    def CheckInitialized(self):
        """Checks for missing or conflicting attributes.

    Subclasses should override this function and raise an exception for
    any errors. Always run this method when all assignments are complete.

    Raises:
      ValidationError: when there are missing or conflicting attributes.
    """

    def ToDict(self):
        """Convert ValidatedBase object to a dictionary.

    Recursively traverses all of its elements and converts everything to
    simplified collections.

    Subclasses should override this method.

    Returns:
      A dictionary mapping all attributes to simple values or collections.
    """
        raise NotImplementedError('Subclasses of ValidatedBase must override ToDict.')

    def ToYAML(self):
        """Print validated object as simplified YAML.

    Returns:
      Object as a simplified YAML string compatible with parsing using the
      SafeLoader.
    """
        return yaml.dump(self.ToDict(), default_flow_style=False, Dumper=ItemDumper)

    def GetWarnings(self):
        """Return all the warnings we've got, along with their associated fields.

    Returns:
      A list of tuples of (dotted_field, warning), both strings.
    """
        raise NotImplementedError('Subclasses of ValidatedBase must override GetWarnings')