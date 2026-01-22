from zope.interface import Attribute, Interface
class IChatService(Interface):
    name = Attribute('A short string identifying this chat service (eg, a hostname)')
    createGroupOnRequest = Attribute('A boolean indicating whether L{getGroup} should implicitly create groups which are requested but which do not yet exist.')
    createUserOnRequest = Attribute('A boolean indicating whether L{getUser} should implicitly create users which are requested but which do not yet exist.')

    def itergroups():
        """Return all groups available on this service.

        @rtype: C{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with a list of C{IGroup} providers.
        """

    def getGroup(name):
        """Retrieve the group by the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the group with the given
        name if one exists (or if one is created due to the setting of
        L{IChatService.createGroupOnRequest}, or which fails with
        L{twisted.words.ewords.NoSuchGroup} if no such group exists.
        """

    def createGroup(name):
        """Create a new group with the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the created group, or
        with fails with L{twisted.words.ewords.DuplicateGroup} if a
        group by that name exists already.
        """

    def lookupGroup(name):
        """Retrieve a group by name.

        Unlike C{getGroup}, this will never implicitly create a group.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the group by the given
        name, or which fails with L{twisted.words.ewords.NoSuchGroup}.
        """

    def getUser(name):
        """Retrieve the user by the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the user with the given
        name if one exists (or if one is created due to the setting of
        L{IChatService.createUserOnRequest}, or which fails with
        L{twisted.words.ewords.NoSuchUser} if no such user exists.
        """

    def createUser(name):
        """Create a new user with the given name.

        @type name: C{str}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with the created user, or
        with fails with L{twisted.words.ewords.DuplicateUser} if a
        user by that name exists already.
        """