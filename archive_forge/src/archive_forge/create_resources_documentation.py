import functools
import json
import jsonschema
import yaml
from ironicclient import exc
Create chassis from dictionaries.

    :param client: ironic client instance.
    :param chassis_list: list of dictionaries to be POSTed to /chassis
        endpoint, if some of them contain "nodes" key, its content is POSTed
        separately to /nodes endpoint.
    :returns: array of exceptions encountered during creation.
    