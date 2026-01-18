from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
def userhostJID(self):
    """
        Extract the bare JID.

        A bare JID does not have a resource part, so this returns a
        L{JID} object representing either C{user@host} or just C{host}.

        If the object this method is called upon doesn't have a resource
        set, it will return itself. Otherwise, the bare JID object will
        be created, interned using L{internJID}.

        @rtype: L{JID}
        """
    if self.resource:
        return internJID(self.userhost())
    else:
        return self