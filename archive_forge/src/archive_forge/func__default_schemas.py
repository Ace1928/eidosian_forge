from __future__ import annotations
import asyncio
import copy
import json
import logging
import typing as t
import warnings
from datetime import datetime, timezone
from jsonschema import ValidationError
from pythonjsonlogger import jsonlogger
from traitlets import Dict, Instance, Set, default
from traitlets.config import Config, LoggingConfigurable
from .schema import SchemaType
from .schema_registry import SchemaRegistry
from .traits import Handlers
from .validators import JUPYTER_EVENTS_CORE_VALIDATOR
@default('schemas')
def _default_schemas(self) -> SchemaRegistry:
    return SchemaRegistry()