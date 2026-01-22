from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
class ContactsList:
    """
    A GUI object that displays a contacts list.

    @ivar chatui: The GUI chat client associated with this contacts list.
    @type chatui: L{ChatUI}

    @ivar contacts: The contacts.
    @type contacts: C{dict} mapping C{str} to a L{IPerson<interfaces.IPerson>}
        provider

    @ivar onlineContacts: The contacts who are currently online (have a status
        that is not C{OFFLINE}).
    @type onlineContacts: C{dict} mapping C{str} to a
        L{IPerson<interfaces.IPerson>} provider

    @ivar clients: The signed-on clients.
    @type clients: C{list} of L{IClient<interfaces.IClient>} providers
    """

    def __init__(self, chatui):
        """
        @param chatui: The GUI chat client associated with this contacts list.
        @type chatui: L{ChatUI}
        """
        self.chatui = chatui
        self.contacts = {}
        self.onlineContacts = {}
        self.clients = []

    def setContactStatus(self, person):
        """
        Inform the user that a person's status has changed.

        @param person: The person whose status has changed.
        @type person: L{IPerson<interfaces.IPerson>} provider
        """
        if person.name not in self.contacts:
            self.contacts[person.name] = person
        if person.name not in self.onlineContacts and (person.status == ONLINE or person.status == AWAY):
            self.onlineContacts[person.name] = person
        if person.name in self.onlineContacts and person.status == OFFLINE:
            del self.onlineContacts[person.name]

    def registerAccountClient(self, client):
        """
        Notify the user that an account client has been signed on to.

        @param client: The client being added to your list of account clients.
        @type client: L{IClient<interfaces.IClient>} provider
        """
        if client not in self.clients:
            self.clients.append(client)

    def unregisterAccountClient(self, client):
        """
        Notify the user that an account client has been signed off or
        disconnected from.

        @param client: The client being removed from the list of account
            clients.
        @type client: L{IClient<interfaces.IClient>} provider
        """
        if client in self.clients:
            self.clients.remove(client)

    def contactChangedNick(self, person, newnick):
        """
        Update your contact information to reflect a change to a contact's
        nickname.

        @param person: The person in your contacts list whose nickname is
            changing.
        @type person: L{IPerson<interfaces.IPerson>} provider

        @param newnick: The new nickname for this person.
        @type newnick: C{str}
        """
        oldname = person.name
        if oldname in self.contacts:
            del self.contacts[oldname]
            person.name = newnick
            self.contacts[newnick] = person
            if oldname in self.onlineContacts:
                del self.onlineContacts[oldname]
                self.onlineContacts[newnick] = person