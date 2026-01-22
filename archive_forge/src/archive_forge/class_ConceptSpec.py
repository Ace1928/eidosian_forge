from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ConceptSpec(object, metaclass=abc.ABCMeta):
    """Base class for concept args."""

    @property
    @abc.abstractmethod
    def attributes(self):
        """A list of Attribute objects representing the attributes of the concept.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """The name of the overall concept."""

    @property
    @abc.abstractmethod
    def anchor(self):
        """The anchor attribute of the concept."""

    @abc.abstractmethod
    def IsAnchor(self, attribute):
        """Returns True if attribute is an anchor."""

    @abc.abstractmethod
    def Initialize(self, fallthroughs_map, parsed_args=None):
        """Initializes the concept using fallthroughs and parsed args."""

    @abc.abstractmethod
    def Parse(self, attribute_to_args_map, base_fallthroughs_map, parsed_args=None, plural=False, allow_empty=False):
        """Lazy parsing function for resource."""

    @abc.abstractmethod
    def BuildFullFallthroughsMap(self, attribute_to_args_map, base_fallthroughs_map):
        """Builds list of fallthroughs for each attribute."""

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.name == other.name and self.attributes == other.attributes

    def __hash__(self):
        return hash(self.name) + hash(self.attributes)