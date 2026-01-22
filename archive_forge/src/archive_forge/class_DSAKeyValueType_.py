import saml2
from saml2 import SamlBase
class DSAKeyValueType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:DSAKeyValueType element"""
    c_tag = 'DSAKeyValueType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}P'] = ('p', P)
    c_cardinality['p'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}Q'] = ('q', Q)
    c_cardinality['q'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}G'] = ('g', G)
    c_cardinality['g'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}Y'] = ('y', Y)
    c_children['{http://www.w3.org/2000/09/xmldsig#}J'] = ('j', J)
    c_cardinality['j'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}Seed'] = ('seed', Seed)
    c_cardinality['seed'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}PgenCounter'] = ('pgen_counter', PgenCounter)
    c_cardinality['pgen_counter'] = {'min': 0, 'max': 1}
    c_child_order.extend(['p', 'q', 'g', 'y', 'j', 'seed', 'pgen_counter'])

    def __init__(self, p=None, q=None, g=None, y=None, j=None, seed=None, pgen_counter=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.p = p
        self.q = q
        self.g = g
        self.y = y
        self.j = j
        self.seed = seed
        self.pgen_counter = pgen_counter