from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
@implementer(iwords.IChatClient)
class IRCUser(irc.IRC):
    """
    Protocol instance representing an IRC user connected to the server.
    """
    groups = None
    logout = None
    avatar = None
    realm = None
    encoding = 'utf-8'

    def connectionMade(self):
        self.irc_PRIVMSG = self.irc_NICKSERV_PRIVMSG
        self.realm = self.factory.realm
        self.hostname = self.realm.name

    def connectionLost(self, reason):
        if self.logout is not None:
            self.logout()
            self.avatar = None

    def sendMessage(self, command, *parameter_list, **kw):
        if 'prefix' not in kw:
            kw['prefix'] = self.hostname
        if 'to' not in kw:
            kw['to'] = self.name.encode(self.encoding)
        arglist = [self, command, kw['to']] + list(parameter_list)
        arglistUnicode = []
        for arg in arglist:
            if isinstance(arg, bytes):
                arg = arg.decode('utf-8')
            arglistUnicode.append(arg)
        irc.IRC.sendMessage(*arglistUnicode, **kw)

    def userJoined(self, group, user):
        self.join(f'{user.name}!{user.name}@{self.hostname}', '#' + group.name)

    def userLeft(self, group, user, reason=None):
        self.part(f'{user.name}!{user.name}@{self.hostname}', '#' + group.name, reason or 'leaving')

    def receive(self, sender, recipient, message):
        if iwords.IGroup.providedBy(recipient):
            recipientName = '#' + recipient.name
        else:
            recipientName = recipient.name
        text = message.get('text', '<an unrepresentable message>')
        for L in text.splitlines():
            self.privmsg(f'{sender.name}!{sender.name}@{self.hostname}', recipientName, L)

    def groupMetaUpdate(self, group, meta):
        if 'topic' in meta:
            topic = meta['topic']
            author = meta.get('topic_author', '')
            self.topic(self.name, '#' + group.name, topic, f'{author}!{author}@{self.hostname}')
    nickname = None
    password = None

    def irc_PASS(self, prefix, params):
        """
        Password message -- Register a password.

        Parameters: <password>

        [REQUIRED]

        Note that IRC requires the client send this *before* NICK
        and USER.
        """
        self.password = params[-1]

    def irc_NICK(self, prefix, params):
        """
        Nick message -- Set your nickname.

        Parameters: <nickname>

        [REQUIRED]
        """
        nickname = params[0]
        try:
            if isinstance(nickname, bytes):
                nickname = nickname.decode(self.encoding)
        except UnicodeDecodeError:
            self.privmsg(NICKSERV, repr(nickname), 'Your nickname cannot be decoded. Please use ASCII or UTF-8.')
            self.transport.loseConnection()
            return
        self.nickname = nickname
        self.name = nickname
        for code, text in self._motdMessages:
            self.sendMessage(code, text % self.factory._serverInfo)
        if self.password is None:
            self.privmsg(NICKSERV, nickname, 'Password?')
        else:
            password = self.password
            self.password = None
            self.logInAs(nickname, password)

    def irc_USER(self, prefix, params):
        """
        User message -- Set your realname.

        Parameters: <user> <mode> <unused> <realname>
        """
        self.realname = params[-1]

    def irc_NICKSERV_PRIVMSG(self, prefix, params):
        """
        Send a (private) message.

        Parameters: <msgtarget> <text to be sent>
        """
        target = params[0]
        password = params[-1]
        if self.nickname is None:
            self.transport.loseConnection()
        elif target.lower() != 'nickserv':
            self.privmsg(NICKSERV, self.nickname, 'Denied.  Please send me (NickServ) your password.')
        else:
            nickname = self.nickname
            self.nickname = None
            self.logInAs(nickname, password)

    def logInAs(self, nickname, password):
        d = self.factory.portal.login(credentials.UsernamePassword(nickname, password), self, iwords.IUser)
        d.addCallbacks(self._cbLogin, self._ebLogin, errbackArgs=(nickname,))
    _welcomeMessages = [(irc.RPL_WELCOME, ':connected to Twisted IRC'), (irc.RPL_YOURHOST, ':Your host is %(serviceName)s, running version %(serviceVersion)s'), (irc.RPL_CREATED, ':This server was created on %(creationDate)s'), (irc.RPL_MYINFO, '%(serviceName)s %(serviceVersion)s w n')]
    _motdMessages = [(irc.RPL_MOTDSTART, ':- %(serviceName)s Message of the Day - '), (irc.RPL_ENDOFMOTD, ':End of /MOTD command.')]

    def _cbLogin(self, result):
        iface, avatar, logout = result
        assert iface is iwords.IUser, f'Realm is buggy, got {iface!r}'
        del self.irc_PRIVMSG
        self.avatar = avatar
        self.logout = logout
        for code, text in self._welcomeMessages:
            self.sendMessage(code, text % self.factory._serverInfo)

    def _ebLogin(self, err, nickname):
        if err.check(ewords.AlreadyLoggedIn):
            self.privmsg(NICKSERV, nickname, 'Already logged in.  No pod people allowed!')
        elif err.check(ecred.UnauthorizedLogin):
            self.privmsg(NICKSERV, nickname, 'Login failed.  Goodbye.')
        else:
            log.msg('Unhandled error during login:')
            log.err(err)
            self.privmsg(NICKSERV, nickname, 'Server error during login.  Sorry.')
        self.transport.loseConnection()

    def irc_PING(self, prefix, params):
        """
        Ping message

        Parameters: <server1> [ <server2> ]
        """
        if self.realm is not None:
            self.sendMessage('PONG', self.hostname)

    def irc_QUIT(self, prefix, params):
        """
        Quit

        Parameters: [ <Quit Message> ]
        """
        self.transport.loseConnection()

    def _channelMode(self, group, modes=None, *args):
        if modes:
            self.sendMessage(irc.ERR_UNKNOWNMODE, ':Unknown MODE flag.')
        else:
            self.channelMode(self.name, '#' + group.name, '+')

    def _userMode(self, user, modes=None):
        if modes:
            self.sendMessage(irc.ERR_UNKNOWNMODE, ':Unknown MODE flag.')
        elif user is self.avatar:
            self.sendMessage(irc.RPL_UMODEIS, '+')
        else:
            self.sendMessage(irc.ERR_USERSDONTMATCH, ":You can't look at someone else's modes.")

    def irc_MODE(self, prefix, params):
        """
        User mode message

        Parameters: <nickname>
        *( ( "+" / "-" ) *( "i" / "w" / "o" / "O" / "r" ) )

        """
        try:
            channelOrUser = params[0]
            if isinstance(channelOrUser, bytes):
                channelOrUser = channelOrUser.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nickname (could not decode your unicode!)')
            return
        if channelOrUser.startswith('#'):

            def ebGroup(err):
                err.trap(ewords.NoSuchGroup)
                self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[0], ":That channel doesn't exist.")
            d = self.realm.lookupGroup(channelOrUser[1:])
            d.addCallbacks(self._channelMode, ebGroup, callbackArgs=tuple(params[1:]))
        else:

            def ebUser(err):
                self.sendMessage(irc.ERR_NOSUCHNICK, ':No such nickname.')
            d = self.realm.lookupUser(channelOrUser)
            d.addCallbacks(self._userMode, ebUser, callbackArgs=tuple(params[1:]))

    def irc_USERHOST(self, prefix, params):
        """
        Userhost message

        Parameters: <nickname> *( SPACE <nickname> )

        [Optional]
        """
        pass

    def irc_PRIVMSG(self, prefix, params):
        """
        Send a (private) message.

        Parameters: <msgtarget> <text to be sent>
        """
        try:
            targetName = params[0]
            if isinstance(targetName, bytes):
                targetName = targetName.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nick/channel (could not decode your unicode!)')
            return
        messageText = params[-1]
        if targetName.startswith('#'):
            target = self.realm.lookupGroup(targetName[1:])
        else:
            target = self.realm.lookupUser(targetName).addCallback(lambda user: user.mind)

        def cbTarget(targ):
            if targ is not None:
                return self.avatar.send(targ, {'text': messageText})

        def ebTarget(err):
            self.sendMessage(irc.ERR_NOSUCHNICK, targetName, ':No such nick/channel.')
        target.addCallbacks(cbTarget, ebTarget)

    def irc_JOIN(self, prefix, params):
        """
        Join message

        Parameters: ( <channel> *( "," <channel> ) [ <key> *( "," <key> ) ] )
        """
        try:
            groupName = params[0]
            if isinstance(groupName, bytes):
                groupName = groupName.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[0], ':No such channel (could not decode your unicode!)')
            return
        if groupName.startswith('#'):
            groupName = groupName[1:]

        def cbGroup(group):

            def cbJoin(ign):
                self.userJoined(group, self)
                self.names(self.name, '#' + group.name, [user.name for user in group.iterusers()])
                self._sendTopic(group)
            return self.avatar.join(group).addCallback(cbJoin)

        def ebGroup(err):
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, '#' + groupName, ':No such channel.')
        self.realm.getGroup(groupName).addCallbacks(cbGroup, ebGroup)

    def irc_PART(self, prefix, params):
        """
        Part message

        Parameters: <channel> *( "," <channel> ) [ <Part Message> ]
        """
        try:
            groupName = params[0]
            if isinstance(params[0], bytes):
                groupName = params[0].decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOTONCHANNEL, params[0], ':Could not decode your unicode!')
            return
        if groupName.startswith('#'):
            groupName = groupName[1:]
        if len(params) > 1:
            reason = params[1]
            if isinstance(reason, bytes):
                reason = reason.decode('utf-8')
        else:
            reason = None

        def cbGroup(group):

            def cbLeave(result):
                self.userLeft(group, self, reason)
            return self.avatar.leave(group, reason).addCallback(cbLeave)

        def ebGroup(err):
            err.trap(ewords.NoSuchGroup)
            self.sendMessage(irc.ERR_NOTONCHANNEL, '#' + groupName, ':' + err.getErrorMessage())
        self.realm.lookupGroup(groupName).addCallbacks(cbGroup, ebGroup)

    def irc_NAMES(self, prefix, params):
        """
        Names message

        Parameters: [ <channel> *( "," <channel> ) [ <target> ] ]
        """
        try:
            channel = params[-1]
            if isinstance(channel, bytes):
                channel = channel.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[-1], ':No such channel (could not decode your unicode!)')
            return
        if channel.startswith('#'):
            channel = channel[1:]

        def cbGroup(group):
            self.names(self.name, '#' + group.name, [user.name for user in group.iterusers()])

        def ebGroup(err):
            err.trap(ewords.NoSuchGroup)
            self.names(self.name, '#' + channel, [])
        self.realm.lookupGroup(channel).addCallbacks(cbGroup, ebGroup)

    def irc_TOPIC(self, prefix, params):
        """
        Topic message

        Parameters: <channel> [ <topic> ]
        """
        try:
            channel = params[0]
            if isinstance(params[0], bytes):
                channel = channel.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, ":That channel doesn't exist (could not decode your unicode!)")
            return
        if channel.startswith('#'):
            channel = channel[1:]
        if len(params) > 1:
            self._setTopic(channel, params[1])
        else:
            self._getTopic(channel)

    def _sendTopic(self, group):
        """
        Send the topic of the given group to this user, if it has one.
        """
        topic = group.meta.get('topic')
        if topic:
            author = group.meta.get('topic_author') or '<noone>'
            date = group.meta.get('topic_date', 0)
            self.topic(self.name, '#' + group.name, topic)
            self.topicAuthor(self.name, '#' + group.name, author, date)

    def _getTopic(self, channel):

        def ebGroup(err):
            err.trap(ewords.NoSuchGroup)
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, '=', channel, ":That channel doesn't exist.")
        self.realm.lookupGroup(channel).addCallbacks(self._sendTopic, ebGroup)

    def _setTopic(self, channel, topic):

        def cbGroup(group):
            newMeta = group.meta.copy()
            newMeta['topic'] = topic
            newMeta['topic_author'] = self.name
            newMeta['topic_date'] = int(time())

            def ebSet(err):
                self.sendMessage(irc.ERR_CHANOPRIVSNEEDED, '#' + group.name, ':You need to be a channel operator to do that.')
            return group.setMetadata(newMeta).addErrback(ebSet)

        def ebGroup(err):
            err.trap(ewords.NoSuchGroup)
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, '=', channel, ":That channel doesn't exist.")
        self.realm.lookupGroup(channel).addCallbacks(cbGroup, ebGroup)

    def list(self, channels):
        """
        Send a group of LIST response lines

        @type channels: C{list} of C{(str, int, str)}
        @param channels: Information about the channels being sent:
            their name, the number of participants, and their topic.
        """
        for name, size, topic in channels:
            self.sendMessage(irc.RPL_LIST, name, str(size), ':' + topic)
        self.sendMessage(irc.RPL_LISTEND, ':End of /LIST')

    def irc_LIST(self, prefix, params):
        """
        List query

        Return information about the indicated channels, or about all
        channels if none are specified.

        Parameters: [ <channel> *( "," <channel> ) [ <target> ] ]
        """
        if params:
            try:
                allChannels = params[0]
                if isinstance(allChannels, bytes):
                    allChannels = allChannels.decode(self.encoding)
                channels = allChannels.split(',')
            except UnicodeDecodeError:
                self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[0], ':No such channel (could not decode your unicode!)')
                return
            groups = []
            for ch in channels:
                if ch.startswith('#'):
                    ch = ch[1:]
                groups.append(self.realm.lookupGroup(ch))
            groups = defer.DeferredList(groups, consumeErrors=True)
            groups.addCallback(lambda gs: [r for s, r in gs if s])
        else:
            groups = self.realm.itergroups()

        def cbGroups(groups):

            def gotSize(size, group):
                return (group.name, size, group.meta.get('topic'))
            d = defer.DeferredList([group.size().addCallback(gotSize, group) for group in groups])
            d.addCallback(lambda results: self.list([r for s, r in results if s]))
            return d
        groups.addCallback(cbGroups)

    def _channelWho(self, group):
        self.who(self.name, '#' + group.name, [(m.name, self.hostname, self.realm.name, m.name, 'H', 0, m.name) for m in group.iterusers()])

    def _userWho(self, user):
        self.sendMessage(irc.RPL_ENDOFWHO, ':User /WHO not implemented')

    def irc_WHO(self, prefix, params):
        """
        Who query

        Parameters: [ <mask> [ "o" ] ]
        """
        if not params:
            self.sendMessage(irc.RPL_ENDOFWHO, ':/WHO not supported.')
            return
        try:
            channelOrUser = params[0]
            if isinstance(channelOrUser, bytes):
                channelOrUser = channelOrUser.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.RPL_ENDOFWHO, params[0], ':End of /WHO list (could not decode your unicode!)')
            return
        if channelOrUser.startswith('#'):

            def ebGroup(err):
                err.trap(ewords.NoSuchGroup)
                self.sendMessage(irc.RPL_ENDOFWHO, channelOrUser, ':End of /WHO list.')
            d = self.realm.lookupGroup(channelOrUser[1:])
            d.addCallbacks(self._channelWho, ebGroup)
        else:

            def ebUser(err):
                err.trap(ewords.NoSuchUser)
                self.sendMessage(irc.RPL_ENDOFWHO, channelOrUser, ':End of /WHO list.')
            d = self.realm.lookupUser(channelOrUser)
            d.addCallbacks(self._userWho, ebUser)

    def irc_WHOIS(self, prefix, params):
        """
        Whois query

        Parameters: [ <target> ] <mask> *( "," <mask> )
        """

        def cbUser(user):
            self.whois(self.name, user.name, user.name, self.realm.name, user.name, self.realm.name, 'Hi mom!', False, int(time() - user.lastMessage), user.signOn, ['#' + group.name for group in user.itergroups()])

        def ebUser(err):
            err.trap(ewords.NoSuchUser)
            self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nick/channel')
        try:
            user = params[0]
            if isinstance(user, bytes):
                user = user.decode(self.encoding)
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nick/channel')
            return
        self.realm.lookupUser(user).addCallbacks(cbUser, ebUser)

    def irc_OPER(self, prefix, params):
        """
        Oper message

        Parameters: <name> <password>
        """
        self.sendMessage(irc.ERR_NOOPERHOST, ':O-lines not applicable')