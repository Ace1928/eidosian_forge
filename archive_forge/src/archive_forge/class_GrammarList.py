from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
class GrammarList(list):
    """A list of Grammar objects containing words or patterns of words that we want the
    recognition service to recognize.

    Example:

    grammar = '#JSGF V1.0; grammar colors; public <color> = aqua | azure | beige | bisque ;'
    grammar_list = GrammarList()
    grammar_list.add_from_string(grammar, 1)

    Wraps the HTML 5 SpeechGrammarList API

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechGrammarList
    """

    def add_from_string(self, src, weight=1.0):
        """
        Takes a src and weight and adds it to the GrammarList as a new
        Grammar object. The new Grammar object is returned.
        """
        grammar = Grammar(src=src, weight=weight)
        self.append(grammar)
        return grammar

    def add_from_uri(self, uri, weight=1.0):
        """
        Takes a grammar present at a specific uri, and adds it to the
        GrammarList as a new Grammar object. The new Grammar object is
        returned.
        """
        grammar = Grammar(uri=uri, weight=weight)
        self.append(grammar)
        return grammar

    def serialize(self):
        """Returns a list of serialized grammars"""
        return [grammar.serialize() for grammar in self]