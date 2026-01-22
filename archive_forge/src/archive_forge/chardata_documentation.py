from __future__ import annotations
import html
import itertools
import re
import unicodedata

    Build a translate mapping that replaces halfwidth and fullwidth forms
    with their standard-width forms.
    