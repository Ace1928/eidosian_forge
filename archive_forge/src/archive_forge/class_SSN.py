from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class SSN(DefinedNamespace):
    """
    Semantic Sensor Network Ontology

    This ontology describes sensors, actuators and observations, and related concepts. It does not describe domain
    concepts, time, locations, etc. these are intended to be included from other ontologies via OWL imports.

    Generated from: http://www.w3.org/ns/ssn/
    Date: 2020-05-26 14:20:09.068204

    """
    Deployment: URIRef
    Input: URIRef
    Output: URIRef
    Property: URIRef
    Stimulus: URIRef
    System: URIRef
    wasOriginatedBy: URIRef
    deployedOnPlatform: URIRef
    deployedSystem: URIRef
    detects: URIRef
    forProperty: URIRef
    hasDeployment: URIRef
    hasInput: URIRef
    hasOutput: URIRef
    hasProperty: URIRef
    hasSubSystem: URIRef
    implementedBy: URIRef
    implements: URIRef
    inDeployment: URIRef
    isPropertyOf: URIRef
    isProxyFor: URIRef
    _NS = Namespace('http://www.w3.org/ns/ssn/')