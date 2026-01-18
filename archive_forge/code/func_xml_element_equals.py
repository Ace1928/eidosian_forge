from lxml import etree
from testtools import matchers
def xml_element_equals(expected_doc, observed_doc):
    """Test whether two XML documents are equivalent.

            This is a recursive algorithm that operates on each element in
            the hierarchy. Siblings are sorted before being checked to
            account for two semantically equivalent documents where siblings
            appear in different document order.

            The sorting algorithm is a little weak in that it could fail for
            documents where siblings at a given level are the same, but have
            different children.

            """
    if expected_doc.tag != observed_doc.tag:
        return False
    if expected_doc.attrib != observed_doc.attrib:
        return False

    def _sorted_children(doc):
        return sorted(doc.getchildren(), key=lambda el: el.tag)
    expected_children = _sorted_children(expected_doc)
    observed_children = _sorted_children(observed_doc)
    if len(expected_children) != len(observed_children):
        return False
    for expected_el, observed_el in zip(expected_children, observed_children):
        if not xml_element_equals(expected_el, observed_el):
            return False
    return True