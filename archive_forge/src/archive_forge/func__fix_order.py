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
def _fix_order(self, jsonld_string, document):
    """
        Fix order of rdfa tags in jsonld string
        by checking the appearance order in the HTML
        """
    json_objects = json.loads(jsonld_string)
    html, head = (document.xpath('/html'), document.xpath('//head'))
    if not html or not head:
        return json_objects
    html_element, head_element = (html[0], head[0])
    values_for_property = defaultdict(list)
    for meta_tag in head_element.xpath('meta[@property]'):
        expanded_property = self._replaceNS(meta_tag.attrib['property'], html_element, head_element)
        values_for_property[expanded_property].append(meta_tag.get('content'))
    for json_object in json_objects:
        keys = json_object.keys()
        for key in keys:
            if type(json_object[key]) is list and len(json_object[key]) > 1:
                self._sort(json_object[key], values_for_property[key])
    return json_objects