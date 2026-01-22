import json
import logging
import re
from collections import defaultdict
from pyRdfa import Options
from pyRdfa import pyRdfa as PyRdfa
from pyRdfa.initialcontext import initial_context
from rdflib import Graph
from rdflib import logger as rdflib_logger  # type: ignore[no-redef]
from extruct.utils import parse_xmldom_html
def _replaceNS(self, prop, html_element, head_element):
    """Expand namespace to match with returned json (e.g.: og -> 'http://ogp.me/ns#')"""
    context = {'owl': 'http://www.w3.org/2002/07/owl#', 'gr': 'http://purl.org/goodrelations/v1#', 'ctag': 'http://commontag.org/ns#', 'cc': 'http://creativecommons.org/ns#', 'grddl': 'http://www.w3.org/2003/g/data-view#', 'rif': 'http://www.w3.org/2007/rif#', 'sioc': 'http://rdfs.org/sioc/ns#', 'skos': 'http://www.w3.org/2004/02/skos/core#', 'xml': 'http://www.w3.org/XML/1998/namespace', 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#', 'rev': 'http://purl.org/stuff/rev#', 'rdfa': 'http://www.w3.org/ns/rdfa#', 'dc': 'http://purl.org/dc/terms/', 'foaf': 'http://xmlns.com/foaf/0.1/', 'void': 'http://rdfs.org/ns/void#', 'ical': 'http://www.w3.org/2002/12/cal/icaltzd#', 'vcard': 'http://www.w3.org/2006/vcard/ns#', 'wdrs': 'http://www.w3.org/2007/05/powder-s#', 'og': 'http://ogp.me/ns#', 'wdr': 'http://www.w3.org/2007/05/powder#', 'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'xhv': 'http://www.w3.org/1999/xhtml/vocab#', 'xsd': 'http://www.w3.org/2001/XMLSchema#', 'v': 'http://rdf.data-vocabulary.org/#', 'skosxl': 'http://www.w3.org/2008/05/skos-xl#', 'schema': 'http://schema.org/'}
    if ':' not in prop:
        return prop
    if 'http://' in prop:
        return prop
    prefix = prop.split(':')[0]
    match = None
    if head_element.get('prefix'):
        match = re.search(prefix + ': [^\\s]+', head_element.get('prefix'))
    if match:
        ns = match.group().split(': ')[1]
        return ns + prop.split(':')[1]
    if 'xmlns:' + prefix in html_element.keys():
        return html_element.get('xmlns:' + prefix) + prop.split(':')[1]
    if prefix in context:
        return context[prefix] + prop.split(':')[1]
    return prop