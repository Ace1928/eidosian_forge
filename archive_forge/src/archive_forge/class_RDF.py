from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class RDF(DefinedNamespace):
    """
    The RDF Concepts Vocabulary (RDF)

    This is the RDF Schema for the RDF vocabulary terms in the RDF Namespace, defined in RDF 1.1 Concepts.

    Generated from: http://www.w3.org/1999/02/22-rdf-syntax-ns#
    Date: 2020-05-26 14:20:05.642859

    dc:date "2019-12-16"

    """
    _fail = True
    _underscore_num = True
    nil: URIRef
    direction: URIRef
    first: URIRef
    language: URIRef
    object: URIRef
    predicate: URIRef
    rest: URIRef
    subject: URIRef
    type: URIRef
    value: URIRef
    Alt: URIRef
    Bag: URIRef
    CompoundLiteral: URIRef
    List: URIRef
    Property: URIRef
    Seq: URIRef
    Statement: URIRef
    HTML: URIRef
    JSON: URIRef
    PlainLiteral: URIRef
    XMLLiteral: URIRef
    langString: URIRef
    _NS = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')