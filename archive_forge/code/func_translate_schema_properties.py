import copy
import json
import jsonpatch
import warlock.model as warlock
def translate_schema_properties(schema_properties):
    """Parse the properties dictionary of a schema document.

    :returns: list of SchemaProperty objects
    """
    properties = []
    for name, prop in schema_properties.items():
        properties.append(SchemaProperty(name, **prop))
    return properties