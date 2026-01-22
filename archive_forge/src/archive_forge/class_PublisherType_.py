import saml2
from saml2 import SamlBase
from saml2 import md
class PublisherType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:PublisherType element"""
    c_tag = 'PublisherType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['PublisherID'] = ('publisher_id', 'md:entityIDType', True)
    c_attributes['CreationInstant'] = ('creation_instant', 'datetime', False)
    c_attributes['SerialNumber'] = ('serial_number', 'string', False)

    def __init__(self, publisher_id=None, creation_instant=None, serial_number=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.publisher_id = publisher_id
        self.creation_instant = creation_instant
        self.serial_number = serial_number