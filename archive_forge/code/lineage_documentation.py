from __future__ import annotations
import json
import logging
import typing as t
from dataclasses import dataclass, field
from sqlglot import Schema, exp, maybe_parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer import Scope, build_scope, find_all_in_scope, normalize_identifiers, qualify
Node to HTML generator using vis.js.

    https://visjs.github.io/vis-network/docs/network/
    