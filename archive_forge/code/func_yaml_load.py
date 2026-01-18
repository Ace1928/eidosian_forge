import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
@classmethod
def yaml_load(cls, constructor, node):
    data = cls()
    yield data
    constructor.construct_mapping(node, data)