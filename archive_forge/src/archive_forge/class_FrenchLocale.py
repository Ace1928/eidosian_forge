import sys
from math import trunc
from typing import (
class FrenchLocale(FrenchBaseLocale, Locale):
    names = ['fr', 'fr-fr']
    month_abbreviations = ['', 'janv', 'févr', 'mars', 'avr', 'mai', 'juin', 'juil', 'août', 'sept', 'oct', 'nov', 'déc']