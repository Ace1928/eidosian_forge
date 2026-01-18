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

        Fix order of rdfa tags in jsonld string
        by checking the appearance order in the HTML
        