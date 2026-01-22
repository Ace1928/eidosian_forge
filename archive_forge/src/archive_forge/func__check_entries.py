from datetime import datetime
import sys
def _check_entries(self, resource_type, entries):
    """Ensure that C{entries} are valid for a C{resource_type} collection.

        @param resource_type: The resource type of the collection the entries
            are in.
        @param entries: A list of dicts representing objects in the
            collection.
        @return: (name, resource_type), where 'name' is the name of the child
            resource type and 'resource_type' is the corresponding resource
            type.
        """
    name, child_resource_type = self._get_child_resource_type(resource_type)
    for entry in entries:
        self._check_resource_type(child_resource_type, entry)
    return (name, child_resource_type)