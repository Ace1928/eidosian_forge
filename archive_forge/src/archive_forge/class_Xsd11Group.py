import warnings
from collections.abc import MutableMapping
from copy import copy as _copy
from typing import TYPE_CHECKING, cast, overload, Any, Iterable, Iterator, \
from xml.etree import ElementTree
from .. import limits
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_SEQUENCE, XSD_ALL, XSD_CHOICE, XSD_ELEMENT, \
from ..aliases import ElementType, NamespacesType, SchemaType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, raw_xml_encode
from ..converters import ElementData
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import ValidationMixin, XsdComponent, XsdType
from .particles import ParticleMixin, OccursCalculator
from .elements import XsdElement, XsdAlternative
from .wildcards import XsdAnyElement, Xsd11AnyElement
from .models import ModelVisitor, iter_unordered_content, iter_collapsed_content
class Xsd11Group(XsdGroup):
    """
    Class for XSD 1.1 *model group* definitions.

    .. The XSD 1.1 model groups differ from XSD 1.0 groups for the 'all' model,
       that can contains also other groups.
    ..  <all
          id = ID
          maxOccurs = (0 | 1) : 1
          minOccurs = (0 | 1) : 1
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (element | any | group)*)
        </all>
    """

    def _parse_content_model(self, content_model: ElementType) -> None:
        self.model = local_name(content_model.tag)
        if self.model == 'all':
            if self.max_occurs not in (0, 1):
                msg = _("maxOccurs must be (0 | 1) for 'all' model groups")
                self.parse_error(msg)
            if self.min_occurs not in (0, 1):
                msg = _("minOccurs must be (0 | 1) for 'all' model groups")
                self.parse_error(msg)
        for child in content_model:
            if child.tag == XSD_ELEMENT:
                self.append(self.schema.xsd_element_class(child, self.schema, self, False))
            elif child.tag == XSD_ANY:
                self._group.append(Xsd11AnyElement(child, self.schema, self))
            elif child.tag in (XSD_SEQUENCE, XSD_CHOICE, XSD_ALL):
                self._group.append(Xsd11Group(child, self.schema, self))
            elif child.tag == XSD_GROUP:
                try:
                    ref = self.schema.resolve_qname(child.attrib['ref'])
                except (KeyError, ValueError, RuntimeError) as err:
                    if 'ref' not in child.attrib:
                        msg = _("missing attribute 'ref' in local group")
                        self.parse_error(msg, child)
                    else:
                        self.parse_error(err, child)
                    continue
                if ref != self.name:
                    xsd_group = Xsd11Group(child, self.schema, self)
                    self._group.append(xsd_group)
                    if (self.model != 'all') ^ (xsd_group.model != 'all'):
                        msg = _('an xs:{0} group cannot include a reference to an xs:{1} group').format(self.model, xsd_group.model)
                        self.parse_error(msg)
                        self.pop()
                elif self.redefine is None:
                    msg = _('Circular definition detected for group %r')
                    self.parse_error(msg % self.name)
                else:
                    if child.get('minOccurs', '1') != '1' or child.get('maxOccurs', '1') != '1':
                        msg = _('Redefined group reference cannot have minOccurs/maxOccurs other than 1')
                        self.parse_error(msg)
                    self._group.append(self.redefine)

    def admits_restriction(self, model: str) -> bool:
        if self.model == model or self.model == 'all':
            return True
        elif self.model == 'choice':
            return model == 'sequence' or len(self.ref or self) <= 1
        else:
            return model == 'choice' or len(self.ref or self) <= 1

    def is_restriction(self, other: ModelParticleType, check_occurs: bool=True) -> bool:
        if not self._group:
            return True
        elif not isinstance(other, ParticleMixin):
            raise XMLSchemaValueError("the argument 'base' must be a %r instance" % ParticleMixin)
        elif not isinstance(other, XsdGroup):
            return self.is_element_restriction(other)
        elif not other:
            return False
        elif len(other) == other.min_occurs == other.max_occurs == 1:
            if len(self) > 1:
                return self.is_restriction(other[0], check_occurs)
            elif self.ref is None and isinstance(self[0], XsdGroup) and self[0].is_pointless(parent=self):
                return self[0].is_restriction(other[0], check_occurs)
        if other.model == 'sequence':
            return self.is_sequence_restriction(other)
        elif other.model == 'all':
            return self.is_all_restriction(other)
        else:
            return self.is_choice_restriction(other)

    def has_occurs_restriction(self, other: Union[ModelParticleType, ParticleMixin, 'OccursCalculator']) -> bool:
        if not isinstance(other, XsdGroup):
            return super().has_occurs_restriction(other)
        elif not self:
            return True
        elif self.effective_min_occurs < other.effective_min_occurs:
            return False
        effective_max_occurs = self.effective_max_occurs
        if effective_max_occurs == 0:
            return True
        elif effective_max_occurs is None:
            return other.effective_max_occurs is None
        try:
            return effective_max_occurs <= other.effective_max_occurs
        except TypeError:
            return True

    def is_sequence_restriction(self, other: XsdGroup) -> bool:
        if not self.has_occurs_restriction(other):
            return False
        check_occurs = other.max_occurs != 0
        item_iterator = iter(self.iter_model())
        item = next(item_iterator, None)
        for other_item in other.iter_model():
            if item is not None and item.is_restriction(other_item, check_occurs):
                item = next(item_iterator, None)
            elif not other_item.is_emptiable():
                break
        else:
            if item is None:
                return True
        item_iterator = iter(self)
        item = next(item_iterator, None)
        for other_item in other.iter_model():
            if item is not None and item.is_restriction(other_item, check_occurs):
                item = next(item_iterator, None)
            elif not other_item.is_emptiable():
                break
        else:
            if item is None:
                return True
        other_items = other.iter_model()
        for other_item in other_items:
            if self.is_restriction(other_item, check_occurs):
                return all((x.is_emptiable() for x in other_items))
            elif not other_item.is_emptiable():
                return False
        else:
            return False

    def is_all_restriction(self, other: XsdGroup) -> bool:
        restriction_items = [x for x in self.iter_model()]
        base_items = [x for x in other.iter_model()]
        wildcards: List[XsdAnyElement] = []
        for w1 in base_items:
            if isinstance(w1, XsdAnyElement):
                for w2 in wildcards:
                    if w1.process_contents == w2.process_contents and w1.occurs == w2.occurs:
                        w2.union(w1)
                        w2.extended = True
                        break
                else:
                    wildcards.append(_copy(w1))
        base_items.extend((w for w in wildcards if hasattr(w, 'extended')))
        if self.model != 'choice':
            restriction_wildcards = [e for e in restriction_items if isinstance(e, XsdAnyElement)]
            for other_item in base_items:
                min_occurs, max_occurs = (0, other_item.max_occurs)
                for k in range(len(restriction_items) - 1, -1, -1):
                    item = restriction_items[k]
                    if item.is_restriction(other_item, check_occurs=False):
                        if max_occurs is None:
                            min_occurs += item.min_occurs
                        elif item.max_occurs is None or max_occurs < item.max_occurs or min_occurs + item.min_occurs > max_occurs:
                            continue
                        else:
                            min_occurs += item.min_occurs
                            max_occurs -= item.max_occurs
                        restriction_items.remove(item)
                        if not min_occurs or max_occurs == 0:
                            break
                else:
                    if self.model == 'all' and restriction_wildcards:
                        if not isinstance(other_item, XsdGroup) and other_item.type and (other_item.type.name != XSD_ANY_TYPE):
                            for w in restriction_wildcards:
                                if w.is_matching(other_item.name, self.target_namespace):
                                    return False
                if min_occurs < other_item.min_occurs:
                    break
            else:
                if not restriction_items:
                    return True
            return False
        not_emptiable_items = {x for x in base_items if x.min_occurs}
        for other_item in base_items:
            min_occurs, max_occurs = (0, other_item.max_occurs)
            for k in range(len(restriction_items) - 1, -1, -1):
                item = restriction_items[k]
                if item.is_restriction(other_item, check_occurs=False):
                    if max_occurs is None:
                        min_occurs += item.min_occurs
                    elif item.max_occurs is None or max_occurs < item.max_occurs or min_occurs + item.min_occurs > max_occurs:
                        continue
                    else:
                        min_occurs += item.min_occurs
                        max_occurs -= item.max_occurs
                    if not_emptiable_items:
                        if len(not_emptiable_items) > 1:
                            continue
                        if other_item not in not_emptiable_items:
                            continue
                    restriction_items.remove(item)
                    if not min_occurs or max_occurs == 0:
                        break
            if min_occurs < other_item.min_occurs:
                break
        else:
            if not restriction_items:
                return True
        if any((not isinstance(x, XsdGroup) for x in restriction_items)):
            return False
        for group in restriction_items:
            if not group.is_restriction(other):
                return False
            for item in not_emptiable_items:
                for e in group:
                    if e.name == item.name:
                        break
                else:
                    return False
        else:
            return True

    def is_choice_restriction(self, other: XsdGroup) -> bool:
        restriction_items = [x for x in self.iter_model()]
        has_not_empty_item = any((e.max_occurs != 0 for e in restriction_items))
        check_occurs = other.max_occurs != 0
        max_occurs: Optional[int] = 0
        other_max_occurs: Optional[int] = 0
        for other_item in other.iter_model():
            for item in restriction_items:
                if other_item is item or item.is_restriction(other_item, check_occurs):
                    if max_occurs is not None:
                        effective_max_occurs = item.effective_max_occurs
                        if effective_max_occurs is None:
                            max_occurs = None
                        elif self.model == 'choice':
                            max_occurs = max(max_occurs, effective_max_occurs)
                        else:
                            max_occurs += effective_max_occurs
                    if other_max_occurs is not None:
                        effective_max_occurs = other_item.effective_max_occurs
                        if effective_max_occurs is None:
                            other_max_occurs = None
                        else:
                            other_max_occurs = max(other_max_occurs, effective_max_occurs)
                    break
                elif item.max_occurs != 0:
                    continue
                elif not other_item.is_matching(item.name):
                    continue
                elif has_not_empty_item:
                    break
                else:
                    return False
            else:
                continue
            restriction_items.remove(item)
        if restriction_items:
            return False
        elif other_max_occurs is None:
            if other.max_occurs != 0:
                return True
            other_max_occurs = 0
        elif other.max_occurs is None:
            if other_max_occurs != 0:
                return True
            other_max_occurs = 0
        else:
            other_max_occurs *= other.max_occurs
        if max_occurs is None:
            return self.max_occurs == 0
        elif self.max_occurs is None:
            return max_occurs == 0
        else:
            return other_max_occurs >= max_occurs * self.max_occurs