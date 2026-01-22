from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef
class XSD(DefinedNamespace):
    """
    W3C XML Schema Definition Language (XSD) 1.1 Part 2: Datatypes

    Generated from: ../schemas/datatypes.xsd
    Date: 2021-09-05 20:37+10

    """
    _NS = Namespace('http://www.w3.org/2001/XMLSchema#')
    ENTITIES: URIRef
    ENTITY: URIRef
    ID: URIRef
    IDREF: URIRef
    IDREFS: URIRef
    NCName: URIRef
    NMTOKEN: URIRef
    NMTOKENS: URIRef
    NOTATION: URIRef
    Name: URIRef
    QName: URIRef
    anyURI: URIRef
    base64Binary: URIRef
    boolean: URIRef
    byte: URIRef
    date: URIRef
    dateTime: URIRef
    dateTimeStamp: URIRef
    dayTimeDuration: URIRef
    decimal: URIRef
    double: URIRef
    duration: URIRef
    float: URIRef
    gDay: URIRef
    gMonth: URIRef
    gMonthDay: URIRef
    gYear: URIRef
    gYearMonth: URIRef
    hexBinary: URIRef
    int: URIRef
    integer: URIRef
    language: URIRef
    long: URIRef
    negativeInteger: URIRef
    nonNegativeInteger: URIRef
    nonPositiveInteger: URIRef
    normalizedString: URIRef
    positiveInteger: URIRef
    short: URIRef
    string: URIRef
    time: URIRef
    token: URIRef
    unsignedByte: URIRef
    unsignedInt: URIRef
    unsignedLong: URIRef
    unsignedShort: URIRef
    yearMonthDuration: URIRef
    ordered: URIRef
    bounded: URIRef
    cardinality: URIRef
    numeric: URIRef
    length: URIRef
    minLength: URIRef
    maxLength: URIRef
    pattern: URIRef
    enumeration: URIRef
    whiteSpace: URIRef
    maxExclusive: URIRef
    maxInclusive: URIRef
    minExclusive: URIRef
    minInclusive: URIRef
    totalDigits: URIRef
    fractionDigits: URIRef
    Assertions: URIRef
    explicitTimezone: URIRef
    year: URIRef
    month: URIRef
    day: URIRef
    hour: URIRef
    minute: URIRef
    second: URIRef
    timezoneOffset: URIRef