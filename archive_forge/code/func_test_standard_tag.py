import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
def test_standard_tag(self):
    round_trip('        !!tag:yaml.org,2002:python/object:map\n        name: Anthon\n        location: Germany\n        language: python\n        ')