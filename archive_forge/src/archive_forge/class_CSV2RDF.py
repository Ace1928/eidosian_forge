from __future__ import annotations
import codecs
import configparser
import csv
import datetime
import fileinput
import getopt
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
import rdflib
from rdflib.namespace import RDF, RDFS, split_uri
from rdflib.term import URIRef
from the headers
class CSV2RDF:

    def __init__(self):
        self.CLASS = None
        self.BASE = None
        self.PROPBASE = None
        self.IDENT = 'auto'
        self.LABEL = None
        self.DEFINECLASS = False
        self.SKIP = 0
        self.DELIM = ','
        self.DEFAULT = None
        self.COLUMNS = {}
        self.PROPS = {}
        self.OUT = sys.stdout
        self.triples = 0

    def triple(self, s, p, o):
        self.OUT.write('%s %s %s .\n' % (s.n3(), p.n3(), o.n3()))
        self.triples += 1

    def convert(self, csvreader):
        start = time.time()
        if self.OUT:
            sys.stderr.write('Output to %s\n' % self.OUT.name)
        if self.IDENT != 'auto' and (not isinstance(self.IDENT, tuple)):
            self.IDENT = (self.IDENT,)
        if not self.BASE:
            warnings.warn('No base given, using http://example.org/instances/')
            self.BASE = rdflib.Namespace('http://example.org/instances/')
        if not self.PROPBASE:
            warnings.warn('No property base given, using http://example.org/property/')
            self.PROPBASE = rdflib.Namespace('http://example.org/props/')
        for x in range(self.SKIP):
            next(csvreader)
        header_labels = list(next(csvreader))
        headers = dict(enumerate([self.PROPBASE[toProperty(x)] for x in header_labels]))
        for k, v in self.PROPS.items():
            headers[k] = v
            header_labels[k] = split_uri(v)[1]
        if self.DEFINECLASS:
            self.triple(self.CLASS, RDF.type, RDFS.Class)
            for i in range(len(headers)):
                h, l_ = (headers[i], header_labels[i])
                if h == '' or l_ == '':
                    continue
                if self.COLUMNS.get(i, self.DEFAULT) == 'ignore':
                    continue
                self.triple(h, RDF.type, RDF.Property)
                self.triple(h, RDFS.label, rdflib.Literal(toPropertyLabel(l_)))
                self.triple(h, RDFS.domain, self.CLASS)
                self.triple(h, RDFS.range, self.COLUMNS.get(i, default_node_make).range())
        rows = 0
        for l_ in csvreader:
            try:
                if self.IDENT == 'auto':
                    uri = self.BASE['%d' % rows]
                else:
                    uri = self.BASE['_'.join([quote(x.encode('utf8').replace(' ', '_'), safe='') for x in index(l_, self.IDENT)])]
                if self.LABEL:
                    self.triple(uri, RDFS.label, rdflib.Literal(' '.join(index(l_, self.LABEL))))
                if self.CLASS:
                    self.triple(uri, RDF.type, self.CLASS)
                for i, x in enumerate(l_):
                    x = x.strip()
                    if x != '':
                        if self.COLUMNS.get(i, self.DEFAULT) == 'ignore':
                            continue
                        try:
                            o = self.COLUMNS.get(i, rdflib.Literal)(x)
                            if isinstance(o, list):
                                for _o in o:
                                    self.triple(uri, headers[i], _o)
                            else:
                                self.triple(uri, headers[i], o)
                        except Exception as e:
                            warnings.warn('Could not process value for column ' + '%d:%s in row %d, ignoring: %s ' % (i, headers[i], rows, e.message))
                rows += 1
                if rows % 100000 == 0:
                    sys.stderr.write('%d rows, %d triples, elapsed %.2fs.\n' % (rows, self.triples, time.time() - start))
            except Exception:
                sys.stderr.write('Error processing line: %d\n' % rows)
                raise
        classes = set()
        for l_, x in uris.items():
            u, c = x
            self.triple(u, RDFS.label, rdflib.Literal(l_))
            if c:
                c = rdflib.URIRef(c)
                classes.add(c)
                self.triple(u, RDF.type, c)
        for c in classes:
            self.triple(c, RDF.type, RDFS.Class)
        self.OUT.close()
        sys.stderr.write('Converted %d rows into %d triples.\n' % (rows, self.triples))
        sys.stderr.write('Took %.2f seconds.\n' % (time.time() - start))