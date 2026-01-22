from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
class ConceptManager(object):
    """A manager that contains all concepts (v2) for a given command.

  This object is responsible for registering all concepts, creating arguments
  for the concepts, and creating a RuntimeParser which will be responsible
  for parsing the concepts.

  Attributes:
    concepts: [base.Concept], a list of concepts.
    runtime_parser: RuntimeParser, the runtime parser manager for all concepts.
  """

    def __init__(self):
        self.concepts = []
        self.runtime_parser = None
        self._command_level_fallthroughs = {}

    def AddConcept(self, concept):
        """Add a single concept.

    This method adds a concept to the ConceptManager. It does not immediately
    have any effect on the command's argparse parser.

    Args:
      concept: base.Concept, an instantiated concept.
    """
        self.concepts.append(concept)

    def AddToParser(self, parser):
        """Adds concept arguments and concept RuntimeParser to argparse parser.

    For each concept, the Attribute() method is called, and all resulting
    attributes and attribute groups are translated into arguments for the
    argparse parser.

    Additionally, a concept-specific RuntimeParser is created with all of the
    resulting attributes from the first step. (This will be responsible for
    parsing the concepts.) It is registered to the argparse parser, and will
    appear in the final parsed namespace under CONCEPT_ARGS.

    Args:
      parser: googlecloudsdk.calliope.parser_arguments.ArgumentInterceptor, the
        argparse parser to which to add argparse arguments.
    """
        attributes = [concept.Attribute() for concept in self.concepts]
        self._AddToArgparse(attributes, parser)
        self.runtime_parser = RuntimeParser(attributes)
        parser.add_concepts(self.runtime_parser)

    def _AddToArgparse(self, attributes, parser):
        """Recursively add an arg definition to the parser."""
        for attribute in attributes:
            if isinstance(attribute, base.Attribute):
                parser.add_argument(attribute.arg_name, **attribute.kwargs)
                continue
            group = parser.add_argument_group(attribute.kwargs.pop('help'), **attribute.kwargs)
            self._AddToArgparse(attribute.attributes, group)