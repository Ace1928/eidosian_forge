from urllib import parse
from cinderclient import api_versions
from cinderclient import base
Update the name and/or description for a group type.

        :param group_type: The ID of the :class:`GroupType` to update.
        :param name: Descriptive name of the group type.
        :param description: Description of the group type.
        :rtype: :class:`GroupType`
        