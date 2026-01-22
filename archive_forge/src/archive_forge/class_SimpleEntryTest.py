from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class SimpleEntryTest(ParserTest, TestCase):
    input_string = u'\n        % maybe the simplest possible\n        % just a comment and one reference\n\n        @ARTICLE{Brett2002marsbar,\n        author = {Matthew Brett and Jean-Luc Anton and Romain Valabregue and Jean-Baptise\n            Poline},\n        title = {{Region of interest analysis using an SPM toolbox}},\n        journal = {Neuroimage},\n        institution = {},\n        year = {2002},\n        volume = {16},\n        pages = {1140--1141},\n        number = {2}\n        }\n    '
    correct_result = BibliographyData({'Brett2002marsbar': Entry('article', fields=[('title', '{Region of interest analysis using an SPM toolbox}'), ('journal', 'Neuroimage'), ('institution', ''), ('year', '2002'), ('volume', '16'), ('pages', '1140--1141'), ('number', '2')], persons={'author': [Person(first='Matthew', last='Brett'), Person(first='Jean-Luc', last='Anton'), Person(first='Romain', last='Valabregue'), Person(first='Jean-Baptise', last='Poline')]})})