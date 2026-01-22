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
class MultitypeResourceInfo(ResourceInfo):
    """ResourceInfo object specifically for multitype resources."""

    def _IsAnchor(self, attribute):
        """Returns true if the attribute is an anchor."""
        return self.concept_spec.IsAnchor(attribute)

    def _GetAnchors(self):
        return [a for a in self.concept_spec.attributes if self._IsAnchor(a)]

    def _IsRequiredArg(self, attribute):
        """Returns True if the attribute arg should be required."""
        anchors = self._GetAnchors()
        return anchors == [attribute] and (not self.fallthroughs_map.get(attribute.name, []))

    def _IsPluralArg(self, attribute):
        return self.concept_spec.Pluralize(attribute, plural=self.plural)

    @property
    def args_required(self):
        """True if resource is required & has a single anchor with no fallthroughs.

    Returns:
      bool, whether the argument group should be required.
    """
        if self.allow_empty:
            return False
        anchors = self._GetAnchors()
        if len(anchors) != 1:
            return False
        anchor = anchors[0]
        if self.fallthroughs_map.get(anchor.name, []):
            return False
        return True

    def GetGroupHelp(self):
        base_text = super(MultitypeResourceInfo, self).GetGroupHelp()
        all_types = [type_.name for type_ in self.concept_spec.type_enum]
        return base_text + ' This resource can be one of the following types: [{}].'.format(', '.join(all_types))

    def _GetHelpTextForAttribute(self, attribute):
        base_text = super(MultitypeResourceInfo, self)._GetHelpTextForAttribute(attribute)
        relevant_types = sorted([type_.name for type_ in self.concept_spec._attribute_to_types_map.get(attribute.name)])
        all_types = [type_.name for type_ in self.concept_spec.type_enum]
        if len(set(relevant_types)) == len(all_types):
            return base_text
        return base_text + ' Must be specified for resource of type {}.'.format(' or '.join(['[{}]'.format(t) for t in relevant_types]))