import sys
from hypothesis import given, strategies
import jsonschema

Fuzzing setup for OSS-Fuzz.

See https://github.com/google/oss-fuzz/tree/master/projects/jsonschema for the
other half of the setup here.
