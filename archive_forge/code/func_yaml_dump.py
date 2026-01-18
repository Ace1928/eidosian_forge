import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
@staticmethod
def yaml_dump(dumper, data):
    return dumper.represent_mapping(u'!xxx', data)