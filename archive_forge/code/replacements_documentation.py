from __future__ import annotations
import logging
import re
from ..token import Token
from .state_core import StateCore
Simple typographic replacements

* ``(c)``, ``(C)`` → ©
* ``(tm)``, ``(TM)`` → ™
* ``(r)``, ``(R)`` → ®
* ``+-`` → ±
* ``...`` → …
* ``?....`` → ?..
* ``!....`` → !..
* ``????????`` → ???
* ``!!!!!`` → !!!
* ``,,,`` → ,
* ``--`` → &ndash
* ``---`` → &mdash
