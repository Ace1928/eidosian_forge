from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class PROF(DefinedNamespace):
    """
    Profiles Vocabulary

    This vocabulary is for describing relationships between standards/specifications, profiles of them and
    supporting artifacts such as validating resources.  This model starts with
    [http://dublincore.org/2012/06/14/dcterms#Standard](dct:Standard) entities which can either be Base
    Specifications (a standard not profiling any other Standard) or Profiles (Standards which do profile others).
    Base Specifications or Profiles can have Resource Descriptors associated with them that defines implementing
    rules for the it. Resource Descriptors must indicate the role they play (to guide, to validate etc.) and the
    formalism they adhere to (dct:format) to allow for content negotiation. A vocabulary of Resource Roles are
    provided alongside this vocabulary but that list is extensible.

    Generated from: https://www.w3.org/ns/dx/prof/profilesont.ttl
    Date: 2020-05-26 14:20:03.542924

    """
    Profile: URIRef
    ResourceDescriptor: URIRef
    ResourceRole: URIRef
    hasToken: URIRef
    hasArtifact: URIRef
    hasResource: URIRef
    hasRole: URIRef
    isInheritedFrom: URIRef
    isProfileOf: URIRef
    isTransitiveProfileOf: URIRef
    _NS = Namespace('http://www.w3.org/ns/dx/prof/')