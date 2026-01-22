import os
import flask
from . import exceptions
from ._utils import AttributeDict
Consolidate the config with priority from high to low provided init
    value > OS environ > default.