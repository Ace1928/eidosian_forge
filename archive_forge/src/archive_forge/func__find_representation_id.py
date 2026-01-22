from datetime import datetime
import sys
def _find_representation_id(self, resource_type, name):
    """Find the WADL XML id for the representation of C{resource_type}.

        Looks in the WADL for the first representiation associated with the
        method for a resource type.

        :return: An XML id (a string).
        """
    get_method = self._get_method(resource_type, name)
    for response in get_method:
        for representation in response:
            representation_url = representation.get('href')
            if representation_url is not None:
                return self._application.lookup_xml_id(representation_url)