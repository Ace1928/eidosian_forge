from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class DC(DefinedNamespace):
    """
    Dublin Core Metadata Element Set, Version 1.1

    Generated from: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_elements.ttl
    Date: 2020-05-26 14:19:58.671906

    """
    _fail = True
    contributor: URIRef
    coverage: URIRef
    creator: URIRef
    date: URIRef
    description: URIRef
    format: URIRef
    identifier: URIRef
    language: URIRef
    publisher: URIRef
    relation: URIRef
    rights: URIRef
    source: URIRef
    subject: URIRef
    title: URIRef
    type: URIRef
    _NS = Namespace('http://purl.org/dc/elements/1.1/')