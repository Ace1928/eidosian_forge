from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_word_boundary(self):
    text = 'The quick ("brown") fox can\'t jump 32.3 feet, right?'
    self.assertEqual(regex.split('(?V1)\\b', text), ['', 'The', ' ', 'quick', ' ("', 'brown', '") ', 'fox', ' ', 'can', "'", 't', ' ', 'jump', ' ', '32', '.', '3', ' ', 'feet', ', ', 'right', '?'])
    self.assertEqual(regex.split('(?V1w)\\b', text), ['', 'The', ' ', 'quick', ' ', '(', '"', 'brown', '"', ')', ' ', 'fox', ' ', "can't", ' ', 'jump', ' ', '32.3', ' ', 'feet', ',', ' ', 'right', '?', ''])
    text = 'The  fox'
    self.assertEqual(regex.split('(?V1)\\b', text), ['', 'The', '  ', 'fox', ''])
    self.assertEqual(regex.split('(?V1w)\\b', text), ['', 'The', '  ', 'fox', ''])
    text = "can't aujourd'hui l'objectif"
    self.assertEqual(regex.split('(?V1)\\b', text), ['', 'can', "'", 't', ' ', 'aujourd', "'", 'hui', ' ', 'l', "'", 'objectif', ''])
    self.assertEqual(regex.split('(?V1w)\\b', text), ['', "can't", ' ', "aujourd'hui", ' ', "l'objectif", ''])