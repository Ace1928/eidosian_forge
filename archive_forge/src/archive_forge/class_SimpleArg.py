from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
class SimpleArg(base.Concept):
    """A basic concept with a single attribute.

  Attributes:
    fallthroughs: [calliope.concepts.deps.Fallthrough], the list of sources of
      data, in priority order, that can provide a value for the attribute if
      not given on the command line. These should only be sources inherent to
      the attribute, such as associated properties, not command- specific
      sources.
    positional: bool, True if the concept is a positional value.
    completer: core.cache.completion_cache.Completer, the completer associated
      with the attribute.
    metavar: string,  a name for the argument in usage messages.
    default: object, the concept value if one is not otherwise specified.
    choices: {name: help}, the possible concept values with help text.
    action: string or argparse.Action, the basic action to take when the
       concept is specified on the command line. Required for the current
       underlying argparse implementation.
  """

    def __init__(self, name, fallthroughs=None, positional=False, completer=None, metavar=None, default=None, choices=None, action=None, **kwargs):
        """Initializes the concept."""
        if name is None:
            raise exceptions.InitializationError('Concept name required.')
        self.fallthroughs = fallthroughs or []
        self.positional = positional
        self.completer = completer
        self.metavar = metavar
        self.default = default
        self.choices = choices
        self.action = action
        super(SimpleArg, self).__init__(name, **kwargs)

    def Attribute(self):
        return base.Attribute(concept=self, fallthroughs=self.fallthroughs, completer=self.completer, metavar=self.metavar, default=self.default, action=self.action, choices=self.choices, **self.MakeArgKwargs())

    def Constraints(self):
        """Returns the type constraints message text if any.

    This message text decribes the Validate() method constraints in English.
    For example, a regex validator could provide prose for a better UX than
    a raw 100 char regex pattern.
    """
        return ''

    def Parse(self, dependencies):
        """Parses the concept.

    Args:
      dependencies: googlecloudsdk.command_lib.concepts.dependency_managers
        .DependencyView, the dependency namespace for the concept.

    Raises:
      exceptions.MissingRequiredArgumentException, if no value is provided and
        one is required.

    Returns:
      str, the value given to the argument.
    """
        try:
            return dependencies.value
        except deps_lib.AttributeNotFoundError as e:
            if self.required:
                raise exceptions.MissingRequiredArgumentError(self.GetPresentationName(), _SubException(e))
            return None

    def GetPresentationName(self):
        """Gets presentation name for the attribute, either positional or flag."""
        if self.positional:
            return names.ConvertToPositionalName(self.name)
        return names.ConvertToFlagName(self.name)

    def IsArgRequired(self):
        """Determines whether command line argument for attribute is required.

    Returns:
      bool: True, if the command line argument is required to be provided,
        meaning that the attribute is required and that there are no
        fallthroughs. There may still be a parsing error if the argument isn't
        provided and none of the fallthroughs work.
    """
        return self.required and (not self.fallthroughs)