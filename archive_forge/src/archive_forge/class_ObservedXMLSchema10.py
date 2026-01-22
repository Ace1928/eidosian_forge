from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
class ObservedXMLSchema10(XMLSchema10):
    xsd_notation_class = observed_builder(XMLSchema10.xsd_notation_class)
    xsd_complex_type_class = observed_builder(XMLSchema10.xsd_complex_type_class)
    xsd_attribute_class = observed_builder(XMLSchema10.xsd_attribute_class)
    xsd_any_attribute_class = observed_builder(XMLSchema10.xsd_any_attribute_class)
    xsd_attribute_group_class = observed_builder(XMLSchema10.xsd_attribute_group_class)
    xsd_group_class = observed_builder(XMLSchema10.xsd_group_class)
    xsd_element_class = observed_builder(XMLSchema10.xsd_element_class)
    xsd_any_class = observed_builder(XMLSchema10.xsd_any_class)
    xsd_atomic_restriction_class = observed_builder(XMLSchema10.xsd_atomic_restriction_class)
    xsd_list_class = observed_builder(XMLSchema10.xsd_list_class)
    xsd_union_class = observed_builder(XMLSchema10.xsd_union_class)
    xsd_key_class = observed_builder(XMLSchema10.xsd_key_class)
    xsd_keyref_class = observed_builder(XMLSchema10.xsd_keyref_class)
    xsd_unique_class = observed_builder(XMLSchema10.xsd_unique_class)