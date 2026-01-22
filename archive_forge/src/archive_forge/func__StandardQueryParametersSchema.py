import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def _StandardQueryParametersSchema(discovery_doc):
    """Sets up dict of standard query parameters."""
    standard_query_schema = {'id': 'StandardQueryParameters', 'type': 'object', 'description': 'Query parameters accepted by all methods.', 'properties': discovery_doc.get('parameters', {})}
    standard_query_schema['properties']['trace'] = {'type': 'string', 'description': 'A tracing token of the form "token:<tokenid>" to include in api requests.', 'location': 'query'}
    return standard_query_schema