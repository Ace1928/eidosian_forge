from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
Modifies properties of an existing zone. If any parameter is None,

        then that parameter will be skipped and will not be taken into
        consideration.

        :param zone_name: string representing the name of the zone.
        :param allow_update:
            0 = No updates allowed.
            1 = Zone accepts both secure and nonsecure updates.
            2 = Zone accepts secure updates only.
        :param disable_wins: Indicates whether the WINS record is replicated.
            If set to TRUE, WINS record replication is disabled.
        :param notify:
            0 = Do not notify secondaries
            1 = Notify Servers listed on the Name Servers Tab
            2 = Notify the specified servers
        :param reverse: Indicates whether the Zone is reverse (TRUE)
            or forward (FALSE).
        :param secure_secondaries:
            0 = Allowed to Any host
            1 = Only to the Servers listed on the Name Servers tab
            2 = To the following servers (destination servers IP addresses
                are specified in SecondaryServers value)
            3 = Zone transfers not allowed
        