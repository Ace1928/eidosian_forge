from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class DCAT(DefinedNamespace):
    """
    The data catalog vocabulary

    DCAT is an RDF vocabulary designed to facilitate interoperability between data catalogs published on the Web.
    By using DCAT to describe datasets in data catalogs, publishers increase discoverability and enable
    applications easily to consume metadata from multiple catalogs. It further enables decentralized publishing of
    catalogs and facilitates federated dataset search across sites. Aggregated DCAT metadata can serve as a
    manifest file to facilitate digital preservation. DCAT is defined at http://www.w3.org/TR/vocab-dcat/. Any
    variance between that normative document and this schema is an error in this schema.

    Generated from: https://www.w3.org/ns/dcat2.ttl
    Date: 2020-05-26 14:19:59.985854

    """
    accessURL: URIRef
    bbox: URIRef
    byteSize: URIRef
    centroid: URIRef
    compressFormat: URIRef
    contactPoint: URIRef
    dataset: URIRef
    distribution: URIRef
    downloadURL: URIRef
    endDate: URIRef
    keyword: URIRef
    landingPage: URIRef
    mediaType: URIRef
    packageFormat: URIRef
    record: URIRef
    startDate: URIRef
    theme: URIRef
    themeTaxonomy: URIRef
    Catalog: URIRef
    CatalogRecord: URIRef
    Dataset: URIRef
    Distribution: URIRef
    DataService: URIRef
    Relationship: URIRef
    Resource: URIRef
    Role: URIRef
    spatialResolutionInMeters: URIRef
    temporalResolution: URIRef
    accessService: URIRef
    catalog: URIRef
    endpointDescription: URIRef
    endpointURL: URIRef
    hadRole: URIRef
    qualifiedRelation: URIRef
    servesDataset: URIRef
    service: URIRef
    _NS = Namespace('http://www.w3.org/ns/dcat#')