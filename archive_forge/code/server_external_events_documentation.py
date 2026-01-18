from novaclient import base
Create one or more server events.

        :param:events: A list of dictionaries containing 'server_uuid', 'name',
                       'status', and 'tag' (which may be absent)
        