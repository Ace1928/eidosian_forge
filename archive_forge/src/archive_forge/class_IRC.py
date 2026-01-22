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
class IRC(protocol.Protocol):
    """
    Internet Relay Chat server protocol.
    """
    buffer = ''
    hostname = None
    encoding: Optional[str] = None

    def connectionMade(self):
        self.channels = []
        if self.hostname is None:
            self.hostname = socket.getfqdn()

    def sendLine(self, line):
        line = line + CR + LF
        if isinstance(line, str):
            useEncoding = self.encoding if self.encoding else 'utf-8'
            line = line.encode(useEncoding)
        self.transport.write(line)

    def sendMessage(self, command, *parameter_list, **prefix):
        """
        Send a line formatted as an IRC message.

        First argument is the command, all subsequent arguments are parameters
        to that command.  If a prefix is desired, it may be specified with the
        keyword argument 'prefix'.

        The L{sendCommand} method is generally preferred over this one.
        Notably, this method does not support sending message tags, while the
        L{sendCommand} method does.
        """
        if not command:
            raise ValueError('IRC message requires a command.')
        if ' ' in command or command[0] == ':':
            raise ValueError("Somebody screwed up, 'cuz this doesn't look like a command to me: %s" % command)
        line = ' '.join([command] + list(parameter_list))
        if 'prefix' in prefix:
            line = ':{} {}'.format(prefix['prefix'], line)
        self.sendLine(line)
        if len(parameter_list) > 15:
            log.msg('Message has %d parameters (RFC allows 15):\n%s' % (len(parameter_list), line))

    def sendCommand(self, command, parameters, prefix=None, tags=None):
        """
        Send to the remote peer a line formatted as an IRC message.

        @param command: The command or numeric to send.
        @type command: L{unicode}

        @param parameters: The parameters to send with the command.
        @type parameters: A L{tuple} or L{list} of L{unicode} parameters

        @param prefix: The prefix to send with the command.  If not
            given, no prefix is sent.
        @type prefix: L{unicode}

        @param tags: A dict of message tags.  If not given, no message
            tags are sent.  The dict key should be the name of the tag
            to send as a string; the value should be the unescaped value
            to send with the tag, or either None or "" if no value is to
            be sent with the tag.
        @type tags: L{dict} of tags (L{unicode}) => values (L{unicode})
        @see: U{https://ircv3.net/specs/core/message-tags-3.2.html}
        """
        if not command:
            raise ValueError('IRC message requires a command.')
        if ' ' in command or command[0] == ':':
            raise ValueError(f'Invalid command: "{command}"')
        if tags is None:
            tags = {}
        line = ' '.join([command] + list(parameters))
        if prefix:
            line = f':{prefix} {line}'
        if tags:
            tagStr = self._stringTags(tags)
            line = f'@{tagStr} {line}'
        self.sendLine(line)
        if len(parameters) > 15:
            log.msg('Message has %d parameters (RFC allows 15):\n%s' % (len(parameters), line))

    def _stringTags(self, tags):
        """
        Converts a tag dictionary to a string.

        @param tags: The tag dict passed to sendMsg.

        @rtype: L{unicode}
        @return: IRCv3-format tag string
        """
        self._validateTags(tags)
        tagStrings = []
        for tag, value in tags.items():
            if value:
                tagStrings.append(f'{tag}={self._escapeTagValue(value)}')
            else:
                tagStrings.append(tag)
        return ';'.join(tagStrings)

    def _validateTags(self, tags):
        """
        Checks the tag dict for errors and raises L{ValueError} if an
        error is found.

        @param tags: The tag dict passed to sendMsg.
        """
        for tag, value in tags.items():
            if not tag:
                raise ValueError('A tag name is required.')
            for char in tag:
                if not char.isalnum() and char not in ('-', '/', '.'):
                    raise ValueError('Tag contains invalid characters.')

    def _escapeTagValue(self, value):
        """
        Escape the given tag value according to U{escaping rules in IRCv3
        <https://ircv3.net/specs/core/message-tags-3.2.html>}.

        @param value: The string value to escape.
        @type value: L{str}

        @return: The escaped string for sending as a message value
        @rtype: L{str}
        """
        return value.replace('\\', '\\\\').replace(';', '\\:').replace(' ', '\\s').replace('\r', '\\r').replace('\n', '\\n')

    def dataReceived(self, data):
        """
        This hack is to support mIRC, which sends LF only, even though the RFC
        says CRLF.  (Also, the flexibility of LineReceiver to turn "line mode"
        on and off was not required.)
        """
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        lines = (self.buffer + data).split(LF)
        self.buffer = lines.pop()
        for line in lines:
            if len(line) <= 2:
                continue
            if line[-1] == CR:
                line = line[:-1]
            prefix, command, params = parsemsg(line)
            command = command.upper()
            self.handleCommand(command, prefix, params)

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

    def irc_unknown(self, prefix, command, params):
        """
        Called by L{handleCommand} on a command that doesn't have a defined
        handler. Subclasses should override this method.
        """
        raise NotImplementedError(command, prefix, params)

    def privmsg(self, sender, recip, message):
        """
        Send a message to a channel or user

        @type sender: C{str} or C{unicode}
        @param sender: Who is sending this message.  Should be of the form
            username!ident@hostmask (unless you know better!).

        @type recip: C{str} or C{unicode}
        @param recip: The recipient of this message.  If a channel, it must
            start with a channel prefix.

        @type message: C{str} or C{unicode}
        @param message: The message being sent.
        """
        self.sendCommand('PRIVMSG', (recip, f':{lowQuote(message)}'), sender)

    def notice(self, sender, recip, message):
        """
        Send a "notice" to a channel or user.

        Notices differ from privmsgs in that the RFC claims they are different.
        Robots are supposed to send notices and not respond to them.  Clients
        typically display notices differently from privmsgs.

        @type sender: C{str} or C{unicode}
        @param sender: Who is sending this message.  Should be of the form
            username!ident@hostmask (unless you know better!).

        @type recip: C{str} or C{unicode}
        @param recip: The recipient of this message.  If a channel, it must
            start with a channel prefix.

        @type message: C{str} or C{unicode}
        @param message: The message being sent.
        """
        self.sendCommand('NOTICE', (recip, f':{message}'), sender)

    def action(self, sender, recip, message):
        """
        Send an action to a channel or user.

        @type sender: C{str} or C{unicode}
        @param sender: Who is sending this message.  Should be of the form
            username!ident@hostmask (unless you know better!).

        @type recip: C{str} or C{unicode}
        @param recip: The recipient of this message.  If a channel, it must
            start with a channel prefix.

        @type message: C{str} or C{unicode}
        @param message: The action being sent.
        """
        self.sendLine(f':{sender} ACTION {recip} :{message}')

    def topic(self, user, channel, topic, author=None):
        """
        Send the topic to a user.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the topic.  Only their nickname, not
            the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this is the topic.

        @type topic: C{str} or C{unicode} or L{None}
        @param topic: The topic string, unquoted, or None if there is no topic.

        @type author: C{str} or C{unicode}
        @param author: If the topic is being changed, the full username and
            hostmask of the person changing it.
        """
        if author is None:
            if topic is None:
                self.sendLine(':%s %s %s %s :%s' % (self.hostname, RPL_NOTOPIC, user, channel, 'No topic is set.'))
            else:
                self.sendLine(':%s %s %s %s :%s' % (self.hostname, RPL_TOPIC, user, channel, lowQuote(topic)))
        else:
            self.sendLine(f':{author} TOPIC {channel} :{lowQuote(topic)}')

    def topicAuthor(self, user, channel, author, date):
        """
        Send the author of and time at which a topic was set for the given
        channel.

        This sends a 333 reply message, which is not part of the IRC RFC.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the topic.  Only their nickname, not
            the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this information is relevant.

        @type author: C{str} or C{unicode}
        @param author: The nickname (without hostmask) of the user who last set
            the topic.

        @type date: C{int}
        @param date: A POSIX timestamp (number of seconds since the epoch) at
            which the topic was last set.
        """
        self.sendLine(':%s %d %s %s %s %d' % (self.hostname, 333, user, channel, author, date))

    def names(self, user, channel, names):
        """
        Send the names of a channel's participants to a user.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the name list.  Only their nickname,
            not the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this is the namelist.

        @type names: C{list} of C{str} or C{unicode}
        @param names: The names to send.
        """
        prefixLength = len(channel) + len(user) + 10
        namesLength = 512 - prefixLength
        L = []
        count = 0
        for n in names:
            if count + len(n) + 1 > namesLength:
                self.sendLine(':%s %s %s = %s :%s' % (self.hostname, RPL_NAMREPLY, user, channel, ' '.join(L)))
                L = [n]
                count = len(n)
            else:
                L.append(n)
                count += len(n) + 1
        if L:
            self.sendLine(':%s %s %s = %s :%s' % (self.hostname, RPL_NAMREPLY, user, channel, ' '.join(L)))
        self.sendLine(':%s %s %s %s :End of /NAMES list' % (self.hostname, RPL_ENDOFNAMES, user, channel))

    def who(self, user, channel, memberInfo):
        """
        Send a list of users participating in a channel.

        @type user: C{str} or C{unicode}
        @param user: The user receiving this member information.  Only their
            nickname, not the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this is the member information.

        @type memberInfo: C{list} of C{tuples}
        @param memberInfo: For each member of the given channel, a 7-tuple
            containing their username, their hostmask, the server to which they
            are connected, their nickname, the letter "H" or "G" (standing for
            "Here" or "Gone"), the hopcount from C{user} to this member, and
            this member's real name.
        """
        for info in memberInfo:
            username, hostmask, server, nickname, flag, hops, realName = info
            assert flag in ('H', 'G')
            self.sendLine(':%s %s %s %s %s %s %s %s %s :%d %s' % (self.hostname, RPL_WHOREPLY, user, channel, username, hostmask, server, nickname, flag, hops, realName))
        self.sendLine(':%s %s %s %s :End of /WHO list.' % (self.hostname, RPL_ENDOFWHO, user, channel))

    def whois(self, user, nick, username, hostname, realName, server, serverInfo, oper, idle, signOn, channels):
        """
        Send information about the state of a particular user.

        @type user: C{str} or C{unicode}
        @param user: The user receiving this information.  Only their nickname,
            not the full hostmask.

        @type nick: C{str} or C{unicode}
        @param nick: The nickname of the user this information describes.

        @type username: C{str} or C{unicode}
        @param username: The user's username (eg, ident response)

        @type hostname: C{str}
        @param hostname: The user's hostmask

        @type realName: C{str} or C{unicode}
        @param realName: The user's real name

        @type server: C{str} or C{unicode}
        @param server: The name of the server to which the user is connected

        @type serverInfo: C{str} or C{unicode}
        @param serverInfo: A descriptive string about that server

        @type oper: C{bool}
        @param oper: Indicates whether the user is an IRC operator

        @type idle: C{int}
        @param idle: The number of seconds since the user last sent a message

        @type signOn: C{int}
        @param signOn: A POSIX timestamp (number of seconds since the epoch)
            indicating the time the user signed on

        @type channels: C{list} of C{str} or C{unicode}
        @param channels: A list of the channels which the user is participating in
        """
        self.sendLine(':%s %s %s %s %s %s * :%s' % (self.hostname, RPL_WHOISUSER, user, nick, username, hostname, realName))
        self.sendLine(':%s %s %s %s %s :%s' % (self.hostname, RPL_WHOISSERVER, user, nick, server, serverInfo))
        if oper:
            self.sendLine(':%s %s %s %s :is an IRC operator' % (self.hostname, RPL_WHOISOPERATOR, user, nick))
        self.sendLine(':%s %s %s %s %d %d :seconds idle, signon time' % (self.hostname, RPL_WHOISIDLE, user, nick, idle, signOn))
        self.sendLine(':%s %s %s %s :%s' % (self.hostname, RPL_WHOISCHANNELS, user, nick, ' '.join(channels)))
        self.sendLine(':%s %s %s %s :End of WHOIS list.' % (self.hostname, RPL_ENDOFWHOIS, user, nick))

    def join(self, who, where):
        """
        Send a join message.

        @type who: C{str} or C{unicode}
        @param who: The name of the user joining.  Should be of the form
            username!ident@hostmask (unless you know better!).

        @type where: C{str} or C{unicode}
        @param where: The channel the user is joining.
        """
        self.sendLine(f':{who} JOIN {where}')

    def part(self, who, where, reason=None):
        """
        Send a part message.

        @type who: C{str} or C{unicode}
        @param who: The name of the user joining.  Should be of the form
            username!ident@hostmask (unless you know better!).

        @type where: C{str} or C{unicode}
        @param where: The channel the user is joining.

        @type reason: C{str} or C{unicode}
        @param reason: A string describing the misery which caused this poor
            soul to depart.
        """
        if reason:
            self.sendLine(f':{who} PART {where} :{reason}')
        else:
            self.sendLine(f':{who} PART {where}')

    def channelMode(self, user, channel, mode, *args):
        """
        Send information about the mode of a channel.

        @type user: C{str} or C{unicode}
        @param user: The user receiving the name list.  Only their nickname,
            not the full hostmask.

        @type channel: C{str} or C{unicode}
        @param channel: The channel for which this is the namelist.

        @type mode: C{str}
        @param mode: A string describing this channel's modes.

        @param args: Any additional arguments required by the modes.
        """
        self.sendLine(':%s %s %s %s %s %s' % (self.hostname, RPL_CHANNELMODEIS, user, channel, mode, ' '.join(args)))