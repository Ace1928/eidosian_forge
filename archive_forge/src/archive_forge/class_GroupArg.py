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
class GroupArg(base.Concept):
    """A group concept.

  Attributes:
    mutex: bool, True if this is a mutex (mutually exclusive) group.
  """

    def __init__(self, name, mutex=False, prefixes=False, **kwargs):
        """Initializes the concept."""
        if name is None:
            raise exceptions.InitializationError('Concept name required.')
        self.mutex = mutex
        self.prefixes = prefixes
        self.concepts = []
        super(GroupArg, self).__init__(name, **kwargs)

    def AddConcept(self, concept):
        new_concept = copy.copy(concept)
        new_concept.name = self._GetSubConceptName(new_concept.name)
        self.concepts.append(new_concept)

    def Attribute(self):
        return base.AttributeGroup(concept=self, attributes=[c.Attribute() for c in self.concepts], mutex=self.mutex, **self.MakeArgKwargs())

    def _GetSubConceptName(self, attribute_name):
        if self.prefixes:
            return names.ConvertToNamespaceName(self.name + '_' + attribute_name)
        return attribute_name

    def Parse(self, dependencies):
        """Returns a namespace with the values of the child concepts."""
        return dependencies

    def GetPresentationName(self):
        """Gets presentation name for the attribute group."""
        return self.name

    def IsArgRequired(self):
        """Determines whether the concept group is required to be specified.

    Returns:
      bool: True, if the command line argument is required to be provided,
        meaning that the attribute is required and that there are no
        fallthroughs. There may still be a parsing error if the argument isn't
        provided and none of the fallthroughs work.
    """
        return self.required and (not any((c.fallthroughs for c in self.concepts)))