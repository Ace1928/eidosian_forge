import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
def test_scalar_00(self):
    round_trip('        Outputs:\n          Vpc:\n            Value: !Ref: vpc    # first tag\n            Export:\n              Name: !Sub "${AWS::StackName}-Vpc"  # second tag\n        ')