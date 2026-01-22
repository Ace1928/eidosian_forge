from copy import copy
import logging
import os.path
import sys
import paramiko
from os_ken import version
from os_ken.lib import hub
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.root import RootCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
class SshServer(paramiko.ServerInterface):
    TERM = 'ansi'
    PROMPT = 'bgpd> '
    WELCOME = '\n\rHello, this is OSKen BGP speaker (version %s).\n\r' % version

    class HelpCmd(Command):
        help_msg = 'show this help'
        command = 'help'

        def action(self, params):
            return self.parent_cmd.question_mark()[0]

    class QuitCmd(Command):
        help_msg = 'exit this session'
        command = 'quit'

        def action(self, params):
            self.api.sshserver.end_session()
            return CommandsResponse(STATUS_OK, True)

    def __init__(self, sock, addr):
        super(SshServer, self).__init__()
        self.sock = sock
        self.addr = addr
        self.is_connected = True
        self.buf = None
        self.chan = None
        self.curpos = None
        self.histindex = None
        self.history = None
        self.prompted = None
        self.promptlen = None
        self.api = InternalApi(log_handler=logging.StreamHandler(sys.stderr))
        setattr(self.api, 'sshserver', self)
        self.root = RootCmd(self.api)
        self.root.subcommands['help'] = self.HelpCmd
        self.root.subcommands['quit'] = self.QuitCmd
        self.transport = paramiko.Transport(self.sock)
        self.transport.load_server_moduli()
        host_key = find_ssh_server_key()
        self.transport.add_server_key(host_key)
        self.transport.start_server(server=self)

    def check_auth_none(self, username):
        return paramiko.AUTH_SUCCESSFUL

    def check_auth_password(self, username, password):
        if username == CONF[SSH_USERNAME] and password == CONF[SSH_PASSWORD]:
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_shell_request(self, channel):
        hub.spawn(self._handle_shell_request)
        return True

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        self.TERM = term
        return True

    def check_channel_window_change_request(self, channel, width, height, pixelwidth, pixelheight):
        return True

    @staticmethod
    def _is_echoable(c):
        return not (c < chr(32) or c == chr(127))

    @staticmethod
    def _is_enter(c):
        return c == chr(13)

    @staticmethod
    def _is_eof(c):
        return c == chr(3)

    @staticmethod
    def _is_esc(c):
        return c == chr(27)

    @staticmethod
    def _is_hist(c):
        return c == chr(16) or c == chr(14)

    @staticmethod
    def _is_del(c):
        return c == chr(4) or c == chr(8) or c == chr(21) or (c == chr(23)) or (c == chr(12)) or (c == chr(127))

    @staticmethod
    def _is_curmov(c):
        return c == chr(1) or c == chr(2) or c == chr(5) or (c == chr(6))

    @staticmethod
    def _is_cmpl(c):
        return c == chr(9)

    def _handle_csi_seq(self):
        c = self.chan.recv(1)
        c = c.decode()
        if c == 'A':
            self._lookup_hist_up()
        elif c == 'B':
            self._lookup_hist_down()
        elif c == 'C':
            self._movcursor(self.curpos + 1)
        elif c == 'D':
            self._movcursor(self.curpos - 1)
        else:
            LOG.error('unknown CSI sequence. do nothing: %c', c)

    def _handle_esc_seq(self):
        c = self.chan.recv(1)
        c = c.decode()
        if c == '[':
            self._handle_csi_seq()
        else:
            LOG.error('non CSI sequence. do nothing')

    def _send_csi_seq(self, cmd):
        self.chan.send('\x1b[' + cmd)

    def _movcursor(self, curpos):
        if self.prompted and curpos < len(self.PROMPT):
            self.curpos = len(self.PROMPT)
        elif self.prompted and curpos > len(self.PROMPT) + len(self.buf):
            self.curpos = len(self.PROMPT) + len(self.buf)
        else:
            self._send_csi_seq('%dG' % (curpos + 1))
            self.curpos = curpos

    def _clearscreen(self, prompt=None):
        if not prompt and self.prompted:
            prompt = self.PROMPT
        self._send_csi_seq('2J')
        self._send_csi_seq('d')
        self._refreshline(prompt=prompt)

    def _clearline(self, prompt=None):
        if not prompt and self.prompted:
            prompt = self.PROMPT
        self.prompted = False
        self._movcursor(0)
        self._send_csi_seq('2K')
        if prompt:
            self.prompted = True
            self.chan.send(prompt)
            self._movcursor(len(prompt))
        self.buf = []

    def _refreshline(self, prompt=None):
        if not prompt and self.prompted:
            prompt = self.PROMPT
        buf = copy(self.buf)
        curpos = copy(self.curpos)
        self._clearline(prompt=prompt)
        self.chan.send(''.join(buf))
        self.buf = buf
        self.curpos = curpos
        self._movcursor(curpos)

    def _refreshnewline(self, prompt=None):
        if not prompt and self.prompted:
            prompt = self.PROMPT
        buf = copy(self.buf)
        curpos = copy(self.curpos)
        self._startnewline(prompt)
        self.chan.send(''.join(buf))
        self.buf = buf
        self.curpos = curpos
        self._movcursor(curpos)

    def _startnewline(self, prompt=None, buf=None):
        buf = buf or []
        if not prompt and self.prompted:
            prompt = self.PROMPT
        if isinstance(buf, str):
            buf = list(buf)
        if self.chan:
            self.buf = buf
            if prompt:
                self.chan.send('\n\r' + prompt + ''.join(buf))
                self.curpos = len(prompt) + len(buf)
                self.prompted = True
            else:
                self.chan.send('\n\r' + ''.join(buf))
                self.curpos = len(buf)
                self.prompted = False

    def _lookup_hist_up(self):
        if len(self.history) == 0:
            return
        self.buf = self.history[self.histindex]
        self.curpos = self.promptlen + len(self.buf)
        self._refreshline()
        if self.histindex + 1 < len(self.history):
            self.histindex += 1

    def _lookup_hist_down(self):
        if self.histindex > 0:
            self.histindex -= 1
            self.buf = self.history[self.histindex]
            self.curpos = self.promptlen + len(self.buf)
            self._refreshline()
        else:
            self._clearline()

    def _do_cmpl(self, buf, is_exec=False):
        cmpleter = self.root
        is_spaced = buf[-1] == ' ' if len(buf) > 0 else False
        cmds = [tkn.strip() for tkn in ''.join(buf).split()]
        ret = []
        for i, cmd in enumerate(cmds):
            subcmds = cmpleter.subcommands
            matches = [x for x in subcmds.keys() if x.startswith(cmd)]
            if len(matches) == 1:
                cmpled_cmd = matches[0]
                cmpleter = subcmds[cmpled_cmd](self.api)
                if is_exec:
                    ret.append(cmpled_cmd)
                    continue
                if i + 1 == len(cmds):
                    if is_spaced:
                        result, cmd = cmpleter('?')
                        result = result.value.replace('\n', '\n\r').rstrip()
                        self.prompted = False
                        buf = copy(buf)
                        self._startnewline(buf=result)
                        self.prompted = True
                        self._startnewline(buf=buf)
                    else:
                        self.buf = buf[:-1 * len(cmd)] + list(cmpled_cmd + ' ')
                        self.curpos += len(cmpled_cmd) - len(cmd) + 1
                        self._refreshline()
            else:
                self.prompted = False
                buf = copy(self.buf)
                if len(matches) == 0:
                    if cmpleter.param_help_msg:
                        self.prompted = True
                        ret.append(cmd)
                        continue
                    else:
                        self._startnewline(buf='Error: Not implemented')
                elif i + 1 < len(cmds):
                    self._startnewline(buf='Error: Ambiguous command')
                else:
                    self._startnewline(buf=', '.join(matches))
                ret = []
                self.prompted = True
                if not is_exec:
                    self._startnewline(buf=buf)
                break
        return ret

    def _execute_cmd(self, cmds):
        result, _ = self.root(cmds)
        LOG.debug('result: %s', result)
        if cmds[0] == 'quit':
            self.is_connected = False
            return result.status
        self.prompted = False
        self._startnewline()
        output = result.value.replace('\n', '\n\r').rstrip()
        self.chan.send(output)
        self.prompted = True
        self._startnewline()
        return result.status

    def end_session(self):
        self._startnewline(prompt=False, buf='bye.\n\r')
        self.chan.close()

    def _handle_shell_request(self):
        LOG.info('session start')
        chan = self.transport.accept(20)
        if not chan:
            LOG.info('transport.accept timed out')
            return
        self.chan = chan
        self.buf = []
        self.curpos = 0
        self.history = []
        self.histindex = 0
        self.prompted = True
        self.chan.send(self.WELCOME)
        self._startnewline()
        while self.is_connected:
            c = self.chan.recv(1)
            c = c.decode()
            if len(c) == 0:
                break
            LOG.debug('ord:%d, hex:0x%x', ord(c), ord(c))
            self.promptlen = len(self.PROMPT) if self.prompted else 0
            if c == '?':
                cmpleter = self.root
                cmds = [tkn.strip() for tkn in ''.join(self.buf).split()]
                for i, cmd in enumerate(cmds):
                    subcmds = cmpleter.subcommands
                    matches = [x for x in subcmds.keys() if x.startswith(cmd)]
                    if len(matches) == 1:
                        cmpled_cmd = matches[0]
                        cmpleter = subcmds[cmpled_cmd](self.api)
                result, cmd = cmpleter('?')
                result = result.value.replace('\n', '\n\r').rstrip()
                self.prompted = False
                buf = copy(self.buf)
                self._startnewline(buf=result)
                self.prompted = True
                self._startnewline(buf=buf)
            elif self._is_echoable(c):
                self.buf.insert(self.curpos - self.promptlen, c)
                self.curpos += 1
                self._refreshline()
            elif self._is_esc(c):
                self._handle_esc_seq()
            elif self._is_eof(c):
                self.end_session()
            elif self._is_curmov(c):
                if c == chr(1):
                    self._movcursor(self.promptlen)
                elif c == chr(2):
                    self._movcursor(self.curpos - 1)
                elif c == chr(5):
                    self._movcursor(self.promptlen + len(self.buf))
                elif c == chr(6):
                    self._movcursor(self.curpos + 1)
                else:
                    LOG.error('unknown cursor move cmd.')
                    continue
            elif self._is_hist(c):
                if c == chr(16):
                    self._lookup_hist_up()
                elif c == chr(14):
                    self._lookup_hist_down()
            elif self._is_del(c):
                if c == chr(4):
                    if self.curpos < self.promptlen + len(self.buf):
                        self.buf.pop(self.curpos - self.promptlen)
                        self._refreshline()
                elif c == chr(8) or c == chr(127):
                    if self.curpos > self.promptlen:
                        self.buf.pop(self.curpos - self.promptlen - 1)
                        self.curpos -= 1
                        self._refreshline()
                elif c == chr(21):
                    self._clearline()
                elif c == chr(23):
                    pos = self.curpos - self.promptlen
                    i = pos
                    flag = False
                    for c in reversed(self.buf[:pos]):
                        if flag and c == ' ':
                            break
                        if c != ' ':
                            flag = True
                        i -= 1
                    del self.buf[i:pos]
                    self.curpos = self.promptlen + i
                    self._refreshline()
                elif c == chr(12):
                    self._clearscreen()
            elif self._is_cmpl(c):
                self._do_cmpl(self.buf)
            elif self._is_enter(c):
                if len(''.join(self.buf).strip()) != 0:
                    cmds = self._do_cmpl(self.buf, is_exec=True)
                    if cmds:
                        self.history.insert(0, self.buf)
                        self.histindex = 0
                        self._execute_cmd(cmds)
                    else:
                        LOG.debug('no command is interpreted. just start a new line.')
                        self._startnewline()
                else:
                    LOG.debug('blank buf is detected. just start a new line.')
                    self._startnewline()
            LOG.debug('curpos: %d, buf: %s, prompted: %s', self.curpos, self.buf, self.prompted)
        LOG.info('session end')