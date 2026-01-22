import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class IRCClient(basic.LineReceiver):
    """
    Internet Relay Chat client protocol, with sprinkles.

    In addition to providing an interface for an IRC client protocol,
    this class also contains reasonable implementations of many common
    CTCP methods.

    TODO
    ====
     - Limit the length of messages sent (because the IRC server probably
       does).
     - Add flood protection/rate limiting for my CTCP replies.
     - NickServ cooperation.  (a mix-in?)

    @ivar nickname: Nickname the client will use.
    @ivar password: Password used to log on to the server.  May be L{None}.
    @ivar realname: Supplied to the server during login as the "Real name"
        or "ircname".  May be L{None}.
    @ivar username: Supplied to the server during login as the "User name".
        May be L{None}

    @ivar userinfo: Sent in reply to a C{USERINFO} CTCP query.  If L{None}, no
        USERINFO reply will be sent.
        "This is used to transmit a string which is settable by
        the user (and never should be set by the client)."
    @ivar fingerReply: Sent in reply to a C{FINGER} CTCP query.  If L{None}, no
        FINGER reply will be sent.
    @type fingerReply: Callable or String

    @ivar versionName: CTCP VERSION reply, client name.  If L{None}, no VERSION
        reply will be sent.
    @type versionName: C{str}, or None.
    @ivar versionNum: CTCP VERSION reply, client version.
    @type versionNum: C{str}, or None.
    @ivar versionEnv: CTCP VERSION reply, environment the client is running in.
    @type versionEnv: C{str}, or None.

    @ivar sourceURL: CTCP SOURCE reply, a URL where the source code of this
        client may be found.  If L{None}, no SOURCE reply will be sent.

    @ivar lineRate: Minimum delay between lines sent to the server.  If
        L{None}, no delay will be imposed.
    @type lineRate: Number of Seconds.

    @ivar motd: Either L{None} or, between receipt of I{RPL_MOTDSTART} and
        I{RPL_ENDOFMOTD}, a L{list} of L{str}, each of which is the content
        of an I{RPL_MOTD} message.

    @ivar erroneousNickFallback: Default nickname assigned when an unregistered
        client triggers an C{ERR_ERRONEUSNICKNAME} while trying to register
        with an illegal nickname.
    @type erroneousNickFallback: C{str}

    @ivar _registered: Whether or not the user is registered. It becomes True
        once a welcome has been received from the server.
    @type _registered: C{bool}

    @ivar _attemptedNick: The nickname that will try to get registered. It may
        change if it is illegal or already taken. L{nickname} becomes the
        L{_attemptedNick} that is successfully registered.
    @type _attemptedNick:  C{str}

    @type supported: L{ServerSupportedFeatures}
    @ivar supported: Available ISUPPORT features on the server

    @type hostname: C{str}
    @ivar hostname: Host name of the IRC server the client is connected to.
        Initially the host name is L{None} and later is set to the host name
        from which the I{RPL_WELCOME} message is received.

    @type _heartbeat: L{task.LoopingCall}
    @ivar _heartbeat: Looping call to perform the keepalive by calling
        L{IRCClient._sendHeartbeat} every L{heartbeatInterval} seconds, or
        L{None} if there is no heartbeat.

    @type heartbeatInterval: C{float}
    @ivar heartbeatInterval: Interval, in seconds, to send I{PING} messages to
        the server as a form of keepalive, defaults to 120 seconds. Use L{None}
        to disable the heartbeat.
    """
    hostname = None
    motd = None
    nickname = 'irc'
    password = None
    realname = None
    username = None
    userinfo = None
    fingerReply = None
    versionName = None
    versionNum = None
    versionEnv = None
    sourceURL = 'http://twistedmatrix.com/downloads/'
    dcc_destdir = '.'
    dcc_sessions = None
    performLogin = 1
    lineRate = None
    _queue = None
    _queueEmptying = None
    delimiter = b'\n'
    __pychecker__ = 'unusednames=params,prefix,channel'
    _registered = False
    _attemptedNick = ''
    erroneousNickFallback = 'defaultnick'
    _heartbeat = None
    heartbeatInterval = 120

    def _reallySendLine(self, line):
        quoteLine = lowQuote(line)
        if isinstance(quoteLine, str):
            quoteLine = quoteLine.encode('utf-8')
        quoteLine += b'\r'
        return basic.LineReceiver.sendLine(self, quoteLine)

    def sendLine(self, line):
        if self.lineRate is None:
            self._reallySendLine(line)
        else:
            self._queue.append(line)
            if not self._queueEmptying:
                self._sendLine()

    def _sendLine(self):
        if self._queue:
            self._reallySendLine(self._queue.pop(0))
            self._queueEmptying = reactor.callLater(self.lineRate, self._sendLine)
        else:
            self._queueEmptying = None

    def connectionLost(self, reason):
        basic.LineReceiver.connectionLost(self, reason)
        self.stopHeartbeat()

    def _createHeartbeat(self):
        """
        Create the heartbeat L{LoopingCall}.
        """
        return task.LoopingCall(self._sendHeartbeat)

    def _sendHeartbeat(self):
        """
        Send a I{PING} message to the IRC server as a form of keepalive.
        """
        self.sendLine('PING ' + self.hostname)

    def stopHeartbeat(self):
        """
        Stop sending I{PING} messages to keep the connection to the server
        alive.

        @since: 11.1
        """
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None

    def startHeartbeat(self):
        """
        Start sending I{PING} messages every L{IRCClient.heartbeatInterval}
        seconds to keep the connection to the server alive during periods of no
        activity.

        @since: 11.1
        """
        self.stopHeartbeat()
        if self.heartbeatInterval is None:
            return
        self._heartbeat = self._createHeartbeat()
        self._heartbeat.start(self.heartbeatInterval, now=False)

    def created(self, when):
        """
        Called with creation date information about the server, usually at logon.

        @type when: C{str}
        @param when: A string describing when the server was created, probably.
        """

    def yourHost(self, info):
        """
        Called with daemon information about the server, usually at logon.

        @type info: C{str}
        @param info: A string describing what software the server is running, probably.
        """

    def myInfo(self, servername, version, umodes, cmodes):
        """
        Called with information about the server, usually at logon.

        @type servername: C{str}
        @param servername: The hostname of this server.

        @type version: C{str}
        @param version: A description of what software this server runs.

        @type umodes: C{str}
        @param umodes: All the available user modes.

        @type cmodes: C{str}
        @param cmodes: All the available channel modes.
        """

    def luserClient(self, info):
        """
        Called with information about the number of connections, usually at logon.

        @type info: C{str}
        @param info: A description of the number of clients and servers
        connected to the network, probably.
        """

    def bounce(self, info):
        """
        Called with information about where the client should reconnect.

        @type info: C{str}
        @param info: A plaintext description of the address that should be
        connected to.
        """

    def isupport(self, options):
        """
        Called with various information about what the server supports.

        @type options: C{list} of C{str}
        @param options: Descriptions of features or limits of the server, possibly
        in the form "NAME=VALUE".
        """

    def luserChannels(self, channels):
        """
        Called with the number of channels existent on the server.

        @type channels: C{int}
        """

    def luserOp(self, ops):
        """
        Called with the number of ops logged on to the server.

        @type ops: C{int}
        """

    def luserMe(self, info):
        """
        Called with information about the server connected to.

        @type info: C{str}
        @param info: A plaintext string describing the number of users and servers
        connected to this server.
        """

    def privmsg(self, user, channel, message):
        """
        Called when I have a message from a user to me or a channel.
        """
        pass

    def joined(self, channel):
        """
        Called when I finish joining a channel.

        channel has the starting character (C{'#'}, C{'&'}, C{'!'}, or C{'+'})
        intact.
        """

    def left(self, channel):
        """
        Called when I have left a channel.

        channel has the starting character (C{'#'}, C{'&'}, C{'!'}, or C{'+'})
        intact.
        """

    def noticed(self, user, channel, message):
        """
        Called when I have a notice from a user to me or a channel.

        If the client makes any automated replies, it must not do so in
        response to a NOTICE message, per the RFC::

            The difference between NOTICE and PRIVMSG is that
            automatic replies MUST NEVER be sent in response to a
            NOTICE message. [...] The object of this rule is to avoid
            loops between clients automatically sending something in
            response to something it received.
        """

    def modeChanged(self, user, channel, set, modes, args):
        """
        Called when users or channel's modes are changed.

        @type user: C{str}
        @param user: The user and hostmask which instigated this change.

        @type channel: C{str}
        @param channel: The channel where the modes are changed. If args is
        empty the channel for which the modes are changing. If the changes are
        at server level it could be equal to C{user}.

        @type set: C{bool} or C{int}
        @param set: True if the mode(s) is being added, False if it is being
        removed. If some modes are added and others removed at the same time
        this function will be called twice, the first time with all the added
        modes, the second with the removed ones. (To change this behaviour
        override the irc_MODE method)

        @type modes: C{str}
        @param modes: The mode or modes which are being changed.

        @type args: C{tuple}
        @param args: Any additional information required for the mode
        change.
        """

    def pong(self, user, secs):
        """
        Called with the results of a CTCP PING query.
        """
        pass

    def signedOn(self):
        """
        Called after successfully signing on to the server.
        """
        pass

    def kickedFrom(self, channel, kicker, message):
        """
        Called when I am kicked from a channel.
        """
        pass

    def nickChanged(self, nick):
        """
        Called when my nick has been changed.
        """
        self.nickname = nick

    def userJoined(self, user, channel):
        """
        Called when I see another user joining a channel.
        """
        pass

    def userLeft(self, user, channel):
        """
        Called when I see another user leaving a channel.
        """
        pass

    def userQuit(self, user, quitMessage):
        """
        Called when I see another user disconnect from the network.
        """
        pass

    def userKicked(self, kickee, channel, kicker, message):
        """
        Called when I observe someone else being kicked from a channel.
        """
        pass

    def action(self, user, channel, data):
        """
        Called when I see a user perform an ACTION on a channel.
        """
        pass

    def topicUpdated(self, user, channel, newTopic):
        """
        In channel, user changed the topic to newTopic.

        Also called when first joining a channel.
        """
        pass

    def userRenamed(self, oldname, newname):
        """
        A user changed their name from oldname to newname.
        """
        pass

    def receivedMOTD(self, motd):
        """
        I received a message-of-the-day banner from the server.

        motd is a list of strings, where each string was sent as a separate
        message from the server. To display, you might want to use::

            '\\n'.join(motd)

        to get a nicely formatted string.
        """
        pass

    def join(self, channel, key=None):
        """
        Join a channel.

        @type channel: C{str}
        @param channel: The name of the channel to join. If it has no prefix,
            C{'#'} will be prepended to it.
        @type key: C{str}
        @param key: If specified, the key used to join the channel.
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        if key:
            self.sendLine(f'JOIN {channel} {key}')
        else:
            self.sendLine(f'JOIN {channel}')

    def leave(self, channel, reason=None):
        """
        Leave a channel.

        @type channel: C{str}
        @param channel: The name of the channel to leave. If it has no prefix,
            C{'#'} will be prepended to it.
        @type reason: C{str}
        @param reason: If given, the reason for leaving.
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        if reason:
            self.sendLine(f'PART {channel} :{reason}')
        else:
            self.sendLine(f'PART {channel}')

    def kick(self, channel, user, reason=None):
        """
        Attempt to kick a user from a channel.

        @type channel: C{str}
        @param channel: The name of the channel to kick the user from. If it has
            no prefix, C{'#'} will be prepended to it.
        @type user: C{str}
        @param user: The nick of the user to kick.
        @type reason: C{str}
        @param reason: If given, the reason for kicking the user.
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        if reason:
            self.sendLine(f'KICK {channel} {user} :{reason}')
        else:
            self.sendLine(f'KICK {channel} {user}')
    part = leave

    def invite(self, user, channel):
        """
        Attempt to invite user to channel

        @type user: C{str}
        @param user: The user to invite
        @type channel: C{str}
        @param channel: The channel to invite the user too

        @since: 11.0
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        self.sendLine(f'INVITE {user} {channel}')

    def topic(self, channel, topic=None):
        """
        Attempt to set the topic of the given channel, or ask what it is.

        If topic is None, then I sent a topic query instead of trying to set the
        topic. The server should respond with a TOPIC message containing the
        current topic of the given channel.

        @type channel: C{str}
        @param channel: The name of the channel to change the topic on. If it
            has no prefix, C{'#'} will be prepended to it.
        @type topic: C{str}
        @param topic: If specified, what to set the topic to.
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        if topic != None:
            self.sendLine(f'TOPIC {channel} :{topic}')
        else:
            self.sendLine(f'TOPIC {channel}')

    def mode(self, chan, set, modes, limit=None, user=None, mask=None):
        """
        Change the modes on a user or channel.

        The C{limit}, C{user}, and C{mask} parameters are mutually exclusive.

        @type chan: C{str}
        @param chan: The name of the channel to operate on.
        @type set: C{bool}
        @param set: True to give the user or channel permissions and False to
            remove them.
        @type modes: C{str}
        @param modes: The mode flags to set on the user or channel.
        @type limit: C{int}
        @param limit: In conjunction with the C{'l'} mode flag, limits the
             number of users on the channel.
        @type user: C{str}
        @param user: The user to change the mode on.
        @type mask: C{str}
        @param mask: In conjunction with the C{'b'} mode flag, sets a mask of
            users to be banned from the channel.
        """
        if set:
            line = f'MODE {chan} +{modes}'
        else:
            line = f'MODE {chan} -{modes}'
        if limit is not None:
            line = '%s %d' % (line, limit)
        elif user is not None:
            line = f'{line} {user}'
        elif mask is not None:
            line = f'{line} {mask}'
        self.sendLine(line)

    def say(self, channel, message, length=None):
        """
        Send a message to a channel

        @type channel: C{str}
        @param channel: The channel to say the message on. If it has no prefix,
            C{'#'} will be prepended to it.
        @type message: C{str}
        @param message: The message to say.
        @type length: C{int}
        @param length: The maximum number of octets to send at a time.  This has
            the effect of turning a single call to C{msg()} into multiple
            commands to the server.  This is useful when long messages may be
            sent that would otherwise cause the server to kick us off or
            silently truncate the text we are sending.  If None is passed, the
            entire message is always send in one command.
        """
        if channel[0] not in CHANNEL_PREFIXES:
            channel = '#' + channel
        self.msg(channel, message, length)

    def _safeMaximumLineLength(self, command):
        """
        Estimate a safe maximum line length for the given command.

        This is done by assuming the maximum values for nickname length,
        realname and hostname combined with the command that needs to be sent
        and some guessing. A theoretical maximum value is used because it is
        possible that our nickname, username or hostname changes (on the server
        side) while the length is still being calculated.
        """
        theoretical = ':{}!{}@{} {}'.format('a' * self.supported.getFeature('NICKLEN'), 'b' * 10, 'c' * 63, command)
        fudge = 10
        return MAX_COMMAND_LENGTH - len(theoretical) - fudge

    def _sendMessage(self, msgType, user, message, length=None):
        """
        Send a message or notice to a user or channel.

        The message will be split into multiple commands to the server if:
         - The message contains any newline characters
         - Any span between newline characters is longer than the given
           line-length.

        @param msgType: Whether a PRIVMSG or NOTICE should be sent.
        @type msgType: C{str}

        @param user: Username or channel name to which to direct the
            message.
        @type user: C{str}

        @param message: Text to send.
        @type message: C{str}

        @param length: Maximum number of octets to send in a single
            command, including the IRC protocol framing. If L{None} is given
            then L{IRCClient._safeMaximumLineLength} is used to determine a
            value.
        @type length: C{int}
        """
        fmt = f'{msgType} {user} :'
        if length is None:
            length = self._safeMaximumLineLength(fmt)
        minimumLength = len(fmt) + 2
        if length <= minimumLength:
            raise ValueError('Maximum length must exceed %d for message to %s' % (minimumLength, user))
        for line in split(message, length - minimumLength):
            self.sendLine(fmt + line)

    def msg(self, user, message, length=None):
        """
        Send a message to a user or channel.

        The message will be split into multiple commands to the server if:
         - The message contains any newline characters
         - Any span between newline characters is longer than the given
           line-length.

        @param user: Username or channel name to which to direct the
            message.
        @type user: C{str}

        @param message: Text to send.
        @type message: C{str}

        @param length: Maximum number of octets to send in a single
            command, including the IRC protocol framing. If L{None} is given
            then L{IRCClient._safeMaximumLineLength} is used to determine a
            value.
        @type length: C{int}
        """
        self._sendMessage('PRIVMSG', user, message, length)

    def notice(self, user, message, length=None):
        """
        Send a notice to a user.

        Notices are like normal message, but should never get automated
        replies.

        @type user: C{str}
        @param user: The user to send a notice to.

        @type message: C{str}
        @param message: The contents of the notice to send.

        @param length: Maximum number of octets to send in a single
            command, including the IRC protocol framing. If L{None} is given
            then L{IRCClient._safeMaximumLineLength} is used to determine a
            value.
        @type length: C{int}
        """
        self._sendMessage('NOTICE', user, message, length)

    def away(self, message=''):
        """
        Mark this client as away.

        @type message: C{str}
        @param message: If specified, the away message.
        """
        self.sendLine('AWAY :%s' % message)

    def back(self):
        """
        Clear the away status.
        """
        self.away()

    def whois(self, nickname, server=None):
        """
        Retrieve user information about the given nickname.

        @type nickname: C{str}
        @param nickname: The nickname about which to retrieve information.

        @since: 8.2
        """
        if server is None:
            self.sendLine('WHOIS ' + nickname)
        else:
            self.sendLine(f'WHOIS {server} {nickname}')

    def register(self, nickname, hostname='foo', servername='bar'):
        """
        Login to the server.

        @type nickname: C{str}
        @param nickname: The nickname to register.
        @type hostname: C{str}
        @param hostname: If specified, the hostname to logon as.
        @type servername: C{str}
        @param servername: If specified, the servername to logon as.
        """
        if self.password is not None:
            self.sendLine('PASS %s' % self.password)
        self.setNick(nickname)
        if self.username is None:
            self.username = nickname
        self.sendLine('USER {} {} {} :{}'.format(self.username, hostname, servername, self.realname))

    def setNick(self, nickname):
        """
        Set this client's nickname.

        @type nickname: C{str}
        @param nickname: The nickname to change to.
        """
        self._attemptedNick = nickname
        self.sendLine('NICK %s' % nickname)

    def quit(self, message=''):
        """
        Disconnect from the server

        @type message: C{str}

        @param message: If specified, the message to give when quitting the
            server.
        """
        self.sendLine('QUIT :%s' % message)

    def describe(self, channel, action):
        """
        Strike a pose.

        @type channel: C{str}
        @param channel: The name of the channel to have an action on. If it
            has no prefix, it is sent to the user of that name.
        @type action: C{str}
        @param action: The action to preform.
        @since: 9.0
        """
        self.ctcpMakeQuery(channel, [('ACTION', action)])
    _pings = None
    _MAX_PINGRING = 12

    def ping(self, user, text=None):
        """
        Measure round-trip delay to another IRC client.
        """
        if self._pings is None:
            self._pings = {}
        if text is None:
            chars = string.ascii_letters + string.digits + string.punctuation
            key = ''.join([random.choice(chars) for i in range(12)])
        else:
            key = str(text)
        self._pings[user, key] = time.time()
        self.ctcpMakeQuery(user, [('PING', key)])
        if len(self._pings) > self._MAX_PINGRING:
            byValue = [(v, k) for k, v in self._pings.items()]
            byValue.sort()
            excess = len(self._pings) - self._MAX_PINGRING
            for i in range(excess):
                del self._pings[byValue[i][1]]

    def dccSend(self, user, file):
        """
        This is supposed to send a user a file directly.  This generally
        doesn't work on any client, and this method is included only for
        backwards compatibility and completeness.

        @param user: C{str} representing the user
        @param file: an open file (unknown, since this is not implemented)
        """
        raise NotImplementedError("XXX!!! Help!  I need to bind a socket, have it listen, and tell me its address.  (and stop accepting once we've made a single connection.)")

    def dccResume(self, user, fileName, port, resumePos):
        """
        Send a DCC RESUME request to another user.
        """
        self.ctcpMakeQuery(user, [('DCC', ['RESUME', fileName, port, resumePos])])

    def dccAcceptResume(self, user, fileName, port, resumePos):
        """
        Send a DCC ACCEPT response to clients who have requested a resume.
        """
        self.ctcpMakeQuery(user, [('DCC', ['ACCEPT', fileName, port, resumePos])])

    def irc_ERR_NICKNAMEINUSE(self, prefix, params):
        """
        Called when we try to register or change to a nickname that is already
        taken.
        """
        self._attemptedNick = self.alterCollidedNick(self._attemptedNick)
        self.setNick(self._attemptedNick)

    def alterCollidedNick(self, nickname):
        """
        Generate an altered version of a nickname that caused a collision in an
        effort to create an unused related name for subsequent registration.

        @param nickname: The nickname a user is attempting to register.
        @type nickname: C{str}

        @returns: A string that is in some way different from the nickname.
        @rtype: C{str}
        """
        return nickname + '_'

    def irc_ERR_ERRONEUSNICKNAME(self, prefix, params):
        """
        Called when we try to register or change to an illegal nickname.

        The server should send this reply when the nickname contains any
        disallowed characters.  The bot will stall, waiting for RPL_WELCOME, if
        we don't handle this during sign-on.

        @note: The method uses the spelling I{erroneus}, as it appears in
            the RFC, section 6.1.
        """
        if not self._registered:
            self.setNick(self.erroneousNickFallback)

    def irc_ERR_PASSWDMISMATCH(self, prefix, params):
        """
        Called when the login was incorrect.
        """
        raise IRCPasswordMismatch('Password Incorrect.')

    def irc_RPL_WELCOME(self, prefix, params):
        """
        Called when we have received the welcome from the server.
        """
        self.hostname = prefix
        self._registered = True
        self.nickname = self._attemptedNick
        self.signedOn()
        self.startHeartbeat()

    def irc_JOIN(self, prefix, params):
        """
        Called when a user joins a channel.
        """
        nick = prefix.split('!')[0]
        channel = params[-1]
        if nick == self.nickname:
            self.joined(channel)
        else:
            self.userJoined(nick, channel)

    def irc_PART(self, prefix, params):
        """
        Called when a user leaves a channel.
        """
        nick = prefix.split('!')[0]
        channel = params[0]
        if nick == self.nickname:
            self.left(channel)
        else:
            self.userLeft(nick, channel)

    def irc_QUIT(self, prefix, params):
        """
        Called when a user has quit.
        """
        nick = prefix.split('!')[0]
        self.userQuit(nick, params[0])

    def irc_MODE(self, user, params):
        """
        Parse a server mode change message.
        """
        channel, modes, args = (params[0], params[1], params[2:])
        if modes[0] not in '-+':
            modes = '+' + modes
        if channel == self.nickname:
            paramModes = self.getUserModeParams()
        else:
            paramModes = self.getChannelModeParams()
        try:
            added, removed = parseModes(modes, args, paramModes)
        except IRCBadModes:
            log.err(None, 'An error occurred while parsing the following MODE message: MODE %s' % (' '.join(params),))
        else:
            if added:
                modes, params = zip(*added)
                self.modeChanged(user, channel, True, ''.join(modes), params)
            if removed:
                modes, params = zip(*removed)
                self.modeChanged(user, channel, False, ''.join(modes), params)

    def irc_PING(self, prefix, params):
        """
        Called when some has pinged us.
        """
        self.sendLine('PONG %s' % params[-1])

    def irc_PRIVMSG(self, prefix, params):
        """
        Called when we get a message.
        """
        user = prefix
        channel = params[0]
        message = params[-1]
        if not message:
            return
        if message[0] == X_DELIM:
            m = ctcpExtract(message)
            if m['extended']:
                self.ctcpQuery(user, channel, m['extended'])
            if not m['normal']:
                return
            message = ' '.join(m['normal'])
        self.privmsg(user, channel, message)

    def irc_NOTICE(self, prefix, params):
        """
        Called when a user gets a notice.
        """
        user = prefix
        channel = params[0]
        message = params[-1]
        if message[0] == X_DELIM:
            m = ctcpExtract(message)
            if m['extended']:
                self.ctcpReply(user, channel, m['extended'])
            if not m['normal']:
                return
            message = ' '.join(m['normal'])
        self.noticed(user, channel, message)

    def irc_NICK(self, prefix, params):
        """
        Called when a user changes their nickname.
        """
        nick = prefix.split('!', 1)[0]
        if nick == self.nickname:
            self.nickChanged(params[0])
        else:
            self.userRenamed(nick, params[0])

    def irc_KICK(self, prefix, params):
        """
        Called when a user is kicked from a channel.
        """
        kicker = prefix.split('!')[0]
        channel = params[0]
        kicked = params[1]
        message = params[-1]
        if kicked.lower() == self.nickname.lower():
            self.kickedFrom(channel, kicker, message)
        else:
            self.userKicked(kicked, channel, kicker, message)

    def irc_TOPIC(self, prefix, params):
        """
        Someone in the channel set the topic.
        """
        user = prefix.split('!')[0]
        channel = params[0]
        newtopic = params[1]
        self.topicUpdated(user, channel, newtopic)

    def irc_RPL_TOPIC(self, prefix, params):
        """
        Called when the topic for a channel is initially reported or when it
        subsequently changes.
        """
        user = prefix.split('!')[0]
        channel = params[1]
        newtopic = params[2]
        self.topicUpdated(user, channel, newtopic)

    def irc_RPL_NOTOPIC(self, prefix, params):
        user = prefix.split('!')[0]
        channel = params[1]
        newtopic = ''
        self.topicUpdated(user, channel, newtopic)

    def irc_RPL_MOTDSTART(self, prefix, params):
        if params[-1].startswith('- '):
            params[-1] = params[-1][2:]
        self.motd = [params[-1]]

    def irc_RPL_MOTD(self, prefix, params):
        if params[-1].startswith('- '):
            params[-1] = params[-1][2:]
        if self.motd is None:
            self.motd = []
        self.motd.append(params[-1])

    def irc_RPL_ENDOFMOTD(self, prefix, params):
        """
        I{RPL_ENDOFMOTD} indicates the end of the message of the day
        messages.  Deliver the accumulated lines to C{receivedMOTD}.
        """
        motd = self.motd
        self.motd = None
        self.receivedMOTD(motd)

    def irc_RPL_CREATED(self, prefix, params):
        self.created(params[1])

    def irc_RPL_YOURHOST(self, prefix, params):
        self.yourHost(params[1])

    def irc_RPL_MYINFO(self, prefix, params):
        info = params[1].split(None, 3)
        while len(info) < 4:
            info.append(None)
        self.myInfo(*info)

    def irc_RPL_BOUNCE(self, prefix, params):
        self.bounce(params[1])

    def irc_RPL_ISUPPORT(self, prefix, params):
        args = params[1:-1]
        self.supported.parse(args)
        self.isupport(args)

    def irc_RPL_LUSERCLIENT(self, prefix, params):
        self.luserClient(params[1])

    def irc_RPL_LUSEROP(self, prefix, params):
        try:
            self.luserOp(int(params[1]))
        except ValueError:
            pass

    def irc_RPL_LUSERCHANNELS(self, prefix, params):
        try:
            self.luserChannels(int(params[1]))
        except ValueError:
            pass

    def irc_RPL_LUSERME(self, prefix, params):
        self.luserMe(params[1])

    def irc_unknown(self, prefix, command, params):
        pass

    def ctcpQuery(self, user, channel, messages):
        """
        Dispatch method for any CTCP queries received.

        Duplicated CTCP queries are ignored and no dispatch is
        made. Unrecognized CTCP queries invoke L{IRCClient.ctcpUnknownQuery}.
        """
        seen = set()
        for tag, data in messages:
            method = getattr(self, 'ctcpQuery_%s' % tag, None)
            if tag not in seen:
                if method is not None:
                    method(user, channel, data)
                else:
                    self.ctcpUnknownQuery(user, channel, tag, data)
            seen.add(tag)

    def ctcpUnknownQuery(self, user, channel, tag, data):
        """
        Fallback handler for unrecognized CTCP queries.

        No CTCP I{ERRMSG} reply is made to remove a potential denial of service
        avenue.
        """
        log.msg(f'Unknown CTCP query from {user!r}: {tag!r} {data!r}')

    def ctcpQuery_ACTION(self, user, channel, data):
        self.action(user, channel, data)

    def ctcpQuery_PING(self, user, channel, data):
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('PING', data)])

    def ctcpQuery_FINGER(self, user, channel, data):
        if data is not None:
            self.quirkyMessage(f"Why did {user} send '{data}' with a FINGER query?")
        if not self.fingerReply:
            return
        if callable(self.fingerReply):
            reply = self.fingerReply()
        else:
            reply = str(self.fingerReply)
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('FINGER', reply)])

    def ctcpQuery_VERSION(self, user, channel, data):
        if data is not None:
            self.quirkyMessage(f"Why did {user} send '{data}' with a VERSION query?")
        if self.versionName:
            nick = user.split('!')[0]
            self.ctcpMakeReply(nick, [('VERSION', '%s:%s:%s' % (self.versionName, self.versionNum or '', self.versionEnv or ''))])

    def ctcpQuery_SOURCE(self, user, channel, data):
        if data is not None:
            self.quirkyMessage(f"Why did {user} send '{data}' with a SOURCE query?")
        if self.sourceURL:
            nick = user.split('!')[0]
            self.ctcpMakeReply(nick, [('SOURCE', self.sourceURL), ('SOURCE', None)])

    def ctcpQuery_USERINFO(self, user, channel, data):
        if data is not None:
            self.quirkyMessage(f"Why did {user} send '{data}' with a USERINFO query?")
        if self.userinfo:
            nick = user.split('!')[0]
            self.ctcpMakeReply(nick, [('USERINFO', self.userinfo)])

    def ctcpQuery_CLIENTINFO(self, user, channel, data):
        """
        A master index of what CTCP tags this client knows.

        If no arguments are provided, respond with a list of known tags, sorted
        in alphabetical order.
        If an argument is provided, provide human-readable help on
        the usage of that tag.
        """
        nick = user.split('!')[0]
        if not data:
            names = sorted(reflect.prefixedMethodNames(self.__class__, 'ctcpQuery_'))
            self.ctcpMakeReply(nick, [('CLIENTINFO', ' '.join(names))])
        else:
            args = data.split()
            method = getattr(self, f'ctcpQuery_{args[0]}', None)
            if not method:
                self.ctcpMakeReply(nick, [('ERRMSG', "CLIENTINFO %s :Unknown query '%s'" % (data, args[0]))])
                return
            doc = getattr(method, '__doc__', '')
            self.ctcpMakeReply(nick, [('CLIENTINFO', doc)])

    def ctcpQuery_ERRMSG(self, user, channel, data):
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('ERRMSG', '%s :No error has occurred.' % data)])

    def ctcpQuery_TIME(self, user, channel, data):
        if data is not None:
            self.quirkyMessage(f"Why did {user} send '{data}' with a TIME query?")
        nick = user.split('!')[0]
        self.ctcpMakeReply(nick, [('TIME', ':%s' % time.asctime(time.localtime(time.time())))])

    def ctcpQuery_DCC(self, user, channel, data):
        """
        Initiate a Direct Client Connection

        @param user: The hostmask of the user/client.
        @type user: L{bytes}

        @param channel: The name of the IRC channel.
        @type channel: L{bytes}

        @param data: The DCC request message.
        @type data: L{bytes}
        """
        if not data:
            return
        dcctype = data.split(None, 1)[0].upper()
        handler = getattr(self, 'dcc_' + dcctype, None)
        if handler:
            if self.dcc_sessions is None:
                self.dcc_sessions = []
            data = data[len(dcctype) + 1:]
            handler(user, channel, data)
        else:
            nick = user.split('!')[0]
            self.ctcpMakeReply(nick, [('ERRMSG', f"DCC {data} :Unknown DCC type '{dcctype}'")])
            self.quirkyMessage(f'{user} offered unknown DCC type {dcctype}')

    def dcc_SEND(self, user, channel, data):
        data = shlex.split(data)
        if len(data) < 3:
            raise IRCBadMessage(f'malformed DCC SEND request: {data!r}')
        filename, address, port = data[:3]
        address = dccParseAddress(address)
        try:
            port = int(port)
        except ValueError:
            raise IRCBadMessage(f'Indecipherable port {port!r}')
        size = -1
        if len(data) >= 4:
            try:
                size = int(data[3])
            except ValueError:
                pass
        self.dccDoSend(user, address, port, filename, size, data)

    def dcc_ACCEPT(self, user, channel, data):
        data = shlex.split(data)
        if len(data) < 3:
            raise IRCBadMessage(f'malformed DCC SEND ACCEPT request: {data!r}')
        filename, port, resumePos = data[:3]
        try:
            port = int(port)
            resumePos = int(resumePos)
        except ValueError:
            return
        self.dccDoAcceptResume(user, filename, port, resumePos)

    def dcc_RESUME(self, user, channel, data):
        data = shlex.split(data)
        if len(data) < 3:
            raise IRCBadMessage(f'malformed DCC SEND RESUME request: {data!r}')
        filename, port, resumePos = data[:3]
        try:
            port = int(port)
            resumePos = int(resumePos)
        except ValueError:
            return
        self.dccDoResume(user, filename, port, resumePos)

    def dcc_CHAT(self, user, channel, data):
        data = shlex.split(data)
        if len(data) < 3:
            raise IRCBadMessage(f'malformed DCC CHAT request: {data!r}')
        filename, address, port = data[:3]
        address = dccParseAddress(address)
        try:
            port = int(port)
        except ValueError:
            raise IRCBadMessage(f'Indecipherable port {port!r}')
        self.dccDoChat(user, channel, address, port, data)

    def dccDoSend(self, user, address, port, fileName, size, data):
        """
        Called when I receive a DCC SEND offer from a client.

        By default, I do nothing here.

        @param user: The hostmask of the requesting user.
        @type user: L{bytes}

        @param address: The IP address of the requesting user.
        @type address: L{bytes}

        @param port: An integer representing the port of the requesting user.
        @type port: L{int}

        @param fileName: The name of the file to be transferred.
        @type fileName: L{bytes}

        @param size: The size of the file to be transferred, which may be C{-1}
            if the size of the file was not specified in the DCC SEND request.
        @type size: L{int}

        @param data: A 3-list of [fileName, address, port].
        @type data: L{list}
        """

    def dccDoResume(self, user, file, port, resumePos):
        """
        Called when a client is trying to resume an offered file via DCC send.
        It should be either replied to with a DCC ACCEPT or ignored (default).

        @param user: The hostmask of the user who wants to resume the transfer
            of a file previously offered via DCC send.
        @type user: L{bytes}

        @param file: The name of the file to resume the transfer of.
        @type file: L{bytes}

        @param port: An integer representing the port of the requesting user.
        @type port: L{int}

        @param resumePos: The position in the file from where the transfer
            should resume.
        @type resumePos: L{int}
        """
        pass

    def dccDoAcceptResume(self, user, file, port, resumePos):
        """
        Called when a client has verified and accepted a DCC resume request
        made by us.  By default it will do nothing.

        @param user: The hostmask of the user who has accepted the DCC resume
            request.
        @type user: L{bytes}

        @param file: The name of the file to resume the transfer of.
        @type file: L{bytes}

        @param port: An integer representing the port of the accepting user.
        @type port: L{int}

        @param resumePos: The position in the file from where the transfer
            should resume.
        @type resumePos: L{int}
        """
        pass

    def dccDoChat(self, user, channel, address, port, data):
        pass

    def ctcpMakeReply(self, user, messages):
        """
        Send one or more C{extended messages} as a CTCP reply.

        @type messages: a list of extended messages.  An extended
        message is a (tag, data) tuple, where 'data' may be L{None}.
        """
        self.notice(user, ctcpStringify(messages))

    def ctcpMakeQuery(self, user, messages):
        """
        Send one or more C{extended messages} as a CTCP query.

        @type messages: a list of extended messages.  An extended
        message is a (tag, data) tuple, where 'data' may be L{None}.
        """
        self.msg(user, ctcpStringify(messages))

    def ctcpReply(self, user, channel, messages):
        """
        Dispatch method for any CTCP replies received.
        """
        for m in messages:
            method = getattr(self, 'ctcpReply_%s' % m[0], None)
            if method:
                method(user, channel, m[1])
            else:
                self.ctcpUnknownReply(user, channel, m[0], m[1])

    def ctcpReply_PING(self, user, channel, data):
        nick = user.split('!', 1)[0]
        if not self._pings or (nick, data) not in self._pings:
            raise IRCBadMessage(f'Bogus PING response from {user}: {data}')
        t0 = self._pings[nick, data]
        self.pong(user, time.time() - t0)

    def ctcpUnknownReply(self, user, channel, tag, data):
        """
        Called when a fitting ctcpReply_ method is not found.

        @param user: The hostmask of the user.
        @type user: L{bytes}

        @param channel: The name of the IRC channel.
        @type channel: L{bytes}

        @param tag: The CTCP request tag for which no fitting method is found.
        @type tag: L{bytes}

        @param data: The CTCP message.
        @type data: L{bytes}
        """
        log.msg(f'Unknown CTCP reply from {user}: {tag} {data}\n')

    def badMessage(self, line, excType, excValue, tb):
        """
        When I get a message that's so broken I can't use it.

        @param line: The indecipherable message.
        @type line: L{bytes}

        @param excType: The exception type of the exception raised by the
            message.
        @type excType: L{type}

        @param excValue: The exception parameter of excType or its associated
            value(the second argument to C{raise}).
        @type excValue: L{BaseException}

        @param tb: The Traceback as a traceback object.
        @type tb: L{traceback}
        """
        log.msg(line)
        log.msg(''.join(traceback.format_exception(excType, excValue, tb)))

    def quirkyMessage(self, s):
        """
        This is called when I receive a message which is peculiar, but not
        wholly indecipherable.

        @param s: The peculiar message.
        @type s: L{bytes}
        """
        log.msg(s + '\n')

    def connectionMade(self):
        self.supported = ServerSupportedFeatures()
        self._queue = []
        if self.performLogin:
            self.register(self.nickname)

    def dataReceived(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        data = data.replace(b'\r', b'')
        basic.LineReceiver.dataReceived(self, data)

    def lineReceived(self, line):
        if bytes != str and isinstance(line, bytes):
            line = line.decode('utf-8')
        line = lowDequote(line)
        try:
            prefix, command, params = parsemsg(line)
            if command in numeric_to_symbolic:
                command = numeric_to_symbolic[command]
            self.handleCommand(command, prefix, params)
        except IRCBadMessage:
            self.badMessage(line, *sys.exc_info())

    def getUserModeParams(self):
        """
        Get user modes that require parameters for correct parsing.

        @rtype: C{[str, str]}
        @return: C{[add, remove]}
        """
        return ['', '']

    def getChannelModeParams(self):
        """
        Get channel modes that require parameters for correct parsing.

        @rtype: C{[str, str]}
        @return: C{[add, remove]}
        """
        params = ['', '']
        prefixes = self.supported.getFeature('PREFIX', {})
        params[0] = params[1] = ''.join(prefixes.keys())
        chanmodes = self.supported.getFeature('CHANMODES')
        if chanmodes is not None:
            params[0] += chanmodes.get('addressModes', '')
            params[0] += chanmodes.get('param', '')
            params[1] = params[0]
            params[0] += chanmodes.get('setParam', '')
        return params

    def handleCommand(self, command, prefix, params):
        """
        Determine the function to call for the given command and call it with
        the given arguments.

        @param command: The IRC command to determine the function for.
        @type command: L{bytes}

        @param prefix: The prefix of the IRC message (as returned by
            L{parsemsg}).
        @type prefix: L{bytes}

        @param params: A list of parameters to call the function with.
        @type params: L{list}
        """
        method = getattr(self, 'irc_%s' % command, None)
        try:
            if method is not None:
                method(prefix, params)
            else:
                self.irc_unknown(prefix, command, params)
        except BaseException:
            log.deferr()

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['dcc_sessions'] = None
        dct['_pings'] = None
        return dct