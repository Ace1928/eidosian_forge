from __future__ import annotations
import ast
import json
import os
import re
import sys
import typing as t
import yaml
from yaml.resolver import Resolver
from yaml.constructor import SafeConstructor
from yaml.error import MarkedYAMLError
from yaml.cyaml import CParser
from yamllint import linter
from yamllint.config import YamlLintConfig
Check the given statement for a documentation assignment.