from tempest.lib.cli import base
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests import client
from designateclient.functionaltests import config
def _is_entity_in_list(self, entity, entity_list):
    """Determines if the given entity exists in the given list.

        Uses the id for comparison.

        Certain entities (e.g. zone import, export) cannot be made
        comparable in a list of CLI output results, because the fields
        in a list command can be different from those in a show command.

        """
    return any([entity_record.id == entity.id for entity_record in entity_list])