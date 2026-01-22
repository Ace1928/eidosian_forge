from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class RDFS(DefinedNamespace):
    """
    The RDF Schema vocabulary (RDFS)

    Generated from: http://www.w3.org/2000/01/rdf-schema#
    Date: 2020-05-26 14:20:05.794866

    """
    _fail = True
    comment: URIRef
    domain: URIRef
    isDefinedBy: URIRef
    label: URIRef
    member: URIRef
    range: URIRef
    seeAlso: URIRef
    subClassOf: URIRef
    subPropertyOf: URIRef
    Class: URIRef
    Container: URIRef
    ContainerMembershipProperty: URIRef
    Datatype: URIRef
    Literal: URIRef
    Resource: URIRef
    _NS = Namespace('http://www.w3.org/2000/01/rdf-schema#')