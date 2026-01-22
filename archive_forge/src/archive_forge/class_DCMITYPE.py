from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class DCMITYPE(DefinedNamespace):
    """
    DCMI Type Vocabulary

    Generated from: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_type.ttl
    Date: 2020-05-26 14:19:59.084150

    """
    _fail = True
    Collection: URIRef
    Dataset: URIRef
    Event: URIRef
    Image: URIRef
    InteractiveResource: URIRef
    MovingImage: URIRef
    PhysicalObject: URIRef
    Service: URIRef
    Software: URIRef
    Sound: URIRef
    StillImage: URIRef
    Text: URIRef
    _NS = Namespace('http://purl.org/dc/dcmitype/')