import yaml
from oslo_serialization import jsonutils
Takes a string and returns a dict containing the parsed structure.

    This includes determination of whether the string is using the
    JSON or YAML format.
    