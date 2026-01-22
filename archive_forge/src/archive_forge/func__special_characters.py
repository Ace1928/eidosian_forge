import logging
import sys
import time
import uuid
import pytest
import panel as pn
def _special_characters(term, iterations=1):
    for _ in range(iterations):
        term.write('Danish Characters: æøåÆØÅ\n')
        term.write('Emoji: Python 🐍  Panel ❤️  LOL 😊 \n')
        term.write('Links: https://awesome-panel.org\n')