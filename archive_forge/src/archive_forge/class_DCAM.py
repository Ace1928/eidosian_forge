from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class DCAM(DefinedNamespace):
    """
    Metadata terms for vocabulary description

    Generated from: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_abstract_model.ttl
    Date: 2020-05-26 14:20:00.970966

    """
    _fail = True
    domainIncludes: URIRef
    memberOf: URIRef
    rangeIncludes: URIRef
    VocabularyEncodingScheme: URIRef
    _NS = Namespace('http://purl.org/dc/dcam/')