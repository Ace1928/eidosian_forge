import pytest
from spacy import registry
from spacy.pipeline import EntityRecognizer
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example
Ensure that resetting doc.ents does not change anything