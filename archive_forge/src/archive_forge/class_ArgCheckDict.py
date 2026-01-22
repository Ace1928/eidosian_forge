import unittest
from traits.api import (
class ArgCheckDict(ArgCheckBase):
    value = Dict(Int, Int, {0: 0, 1: 1, 2: 2})