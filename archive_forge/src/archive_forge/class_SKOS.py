from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class SKOS(DefinedNamespace):
    """
    SKOS Vocabulary

    An RDF vocabulary for describing the basic structure and content of concept schemes such as thesauri,
    classification schemes, subject heading lists, taxonomies, 'folksonomies', other types of controlled
    vocabulary, and also concept schemes embedded in glossaries and terminologies.

    Generated from: https://www.w3.org/2009/08/skos-reference/skos.rdf
    Date: 2020-05-26 14:20:08.489187

    """
    _fail = True
    altLabel: URIRef
    broadMatch: URIRef
    broader: URIRef
    broaderTransitive: URIRef
    changeNote: URIRef
    closeMatch: URIRef
    definition: URIRef
    editorialNote: URIRef
    exactMatch: URIRef
    example: URIRef
    hasTopConcept: URIRef
    hiddenLabel: URIRef
    historyNote: URIRef
    inScheme: URIRef
    mappingRelation: URIRef
    member: URIRef
    memberList: URIRef
    narrowMatch: URIRef
    narrower: URIRef
    narrowerTransitive: URIRef
    notation: URIRef
    note: URIRef
    prefLabel: URIRef
    related: URIRef
    relatedMatch: URIRef
    scopeNote: URIRef
    semanticRelation: URIRef
    topConceptOf: URIRef
    Collection: URIRef
    Concept: URIRef
    ConceptScheme: URIRef
    OrderedCollection: URIRef
    _NS = Namespace('http://www.w3.org/2004/02/skos/core#')