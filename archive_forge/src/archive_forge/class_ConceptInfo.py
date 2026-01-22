from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
class ConceptInfo(six.with_metaclass(abc.ABCMeta, object)):
    """Holds information for a concept argument.

  The ConceptInfo object is responsible for holding information about the
  dependencies of a concept, and building a Deps object when it is time for
  lazy parsing of the concept.

  Attributes:
    concept_spec: The concept spec underlying the concept handler.
    attribute_to_args_map: A map of attributes to the names of their associated
      flags.
    fallthroughs_map: A map of attributes to non-argument fallthroughs.
  """

    @property
    def concept_spec(self):
        """The concept spec associated with this info class."""
        raise NotImplementedError

    @property
    def fallthroughs_map(self):
        """A map of attribute names to non-primary fallthroughs."""
        raise NotImplementedError

    @abc.abstractmethod
    def GetHints(self, attribute_name):
        """Get a list of string hints for how to specify a concept's attribute.

    Args:
      attribute_name: str, the name of the attribute to get hints for.

    Returns:
      [str], a list of string hints.
    """

    def GetGroupHelp(self):
        """Get the group help for the group defined by the presentation spec.

    Must be overridden in subclasses.

    Returns:
      (str) the help text.
    """
        raise NotImplementedError

    def GetAttributeArgs(self):
        """Generate args to add to the argument group.

    Must be overridden in subclasses.

    Yields:
      (calliope.base.Argument), all arguments corresponding to concept
        attributes.
    """
        raise NotImplementedError

    def AddToParser(self, parser):
        """Adds all attribute args for the concept to argparse.

    Must be overridden in subclasses.

    Args:
      parser: the parser for the Calliope command.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def Parse(self, parsed_args=None):
        """Lazy parsing function to parse concept.

    Args:
      parsed_args: the argparse namespace from the runtime handler.

    Returns:
      the parsed concept.
    """

    def ClearCache(self):
        """Clear cache if it exists. Override where needed."""
        pass