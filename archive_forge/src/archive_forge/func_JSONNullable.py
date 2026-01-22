import json
import textwrap
def JSONNullable(json_type):
    """Express a JSON schema type as nullable to easily support Parameters that allow_None"""
    return {'anyOf': [json_type, {'type': 'null'}]}