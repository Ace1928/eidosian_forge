from __future__ import annotations
import argparse
import datetime
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple
from rdflib.graph import Graph  # noqa: E402
from rdflib.namespace import DCTERMS, OWL, RDFS, SKOS  # noqa: E402
from rdflib.util import guess_format  # noqa: E402
from rdflib.namespace import DefinedNamespace, Namespace

This rdflib Python script creates a DefinedNamespace Python file from a given RDF file

It is a very simple script: it finds all things defined in the RDF file within a given
namespace:

    <thing> a ?x

    where ?x is anything and <thing> starts with the given namespace

Nicholas J. Car, Dec, 2021
