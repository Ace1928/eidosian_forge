from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode

        Request the statuses of users. Should be sent once after login.

        Args:
            - jids: A list of jids representing the users whose statuses you are
                trying to get.
        