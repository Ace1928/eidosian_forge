import base64
import getpass
import os
import signal
import struct
import sys
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from typing import List, Tuple
from twisted.conch import error
from twisted.conch.client.default import isInKnownHosts
from twisted.conch.ssh import (
from twisted.conch.ui import tkvt100
from twisted.internet import defer, protocol, reactor, tksupport
from twisted.python import log, usage
class TkConchMenu(Tkinter.Frame):

    def __init__(self, *args, **params):
        Tkinter.Frame.__init__(self, *args, **params)
        self.master.title('TkConch')
        self.localRemoteVar = Tkinter.StringVar()
        self.localRemoteVar.set('local')
        Tkinter.Label(self, anchor='w', justify='left', text='Hostname').grid(column=1, row=1, sticky='w')
        self.host = Tkinter.Entry(self)
        self.host.grid(column=2, columnspan=2, row=1, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Port').grid(column=1, row=2, sticky='w')
        self.port = Tkinter.Entry(self)
        self.port.grid(column=2, columnspan=2, row=2, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Username').grid(column=1, row=3, sticky='w')
        self.user = Tkinter.Entry(self)
        self.user.grid(column=2, columnspan=2, row=3, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Command').grid(column=1, row=4, sticky='w')
        self.command = Tkinter.Entry(self)
        self.command.grid(column=2, columnspan=2, row=4, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Identity').grid(column=1, row=5, sticky='w')
        self.identity = Tkinter.Entry(self)
        self.identity.grid(column=2, row=5, sticky='nesw')
        Tkinter.Button(self, command=self.getIdentityFile, text='Browse').grid(column=3, row=5, sticky='nesw')
        Tkinter.Label(self, text='Port Forwarding').grid(column=1, row=6, sticky='w')
        self.forwards = Tkinter.Listbox(self, height=0, width=0)
        self.forwards.grid(column=2, columnspan=2, row=6, sticky='nesw')
        Tkinter.Button(self, text='Add', command=self.addForward).grid(column=1, row=7)
        Tkinter.Button(self, text='Remove', command=self.removeForward).grid(column=1, row=8)
        self.forwardPort = Tkinter.Entry(self)
        self.forwardPort.grid(column=2, row=7, sticky='nesw')
        Tkinter.Label(self, text='Port').grid(column=3, row=7, sticky='nesw')
        self.forwardHost = Tkinter.Entry(self)
        self.forwardHost.grid(column=2, row=8, sticky='nesw')
        Tkinter.Label(self, text='Host').grid(column=3, row=8, sticky='nesw')
        self.localForward = Tkinter.Radiobutton(self, text='Local', variable=self.localRemoteVar, value='local')
        self.localForward.grid(column=2, row=9)
        self.remoteForward = Tkinter.Radiobutton(self, text='Remote', variable=self.localRemoteVar, value='remote')
        self.remoteForward.grid(column=3, row=9)
        Tkinter.Label(self, text='Advanced Options').grid(column=1, columnspan=3, row=10, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Cipher').grid(column=1, row=11, sticky='w')
        self.cipher = Tkinter.Entry(self, name='cipher')
        self.cipher.grid(column=2, columnspan=2, row=11, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='MAC').grid(column=1, row=12, sticky='w')
        self.mac = Tkinter.Entry(self, name='mac')
        self.mac.grid(column=2, columnspan=2, row=12, sticky='nesw')
        Tkinter.Label(self, anchor='w', justify='left', text='Escape Char').grid(column=1, row=13, sticky='w')
        self.escape = Tkinter.Entry(self, name='escape')
        self.escape.grid(column=2, columnspan=2, row=13, sticky='nesw')
        Tkinter.Button(self, text='Connect!', command=self.doConnect).grid(column=1, columnspan=3, row=14, sticky='nesw')
        self.grid_rowconfigure(6, weight=1, minsize=64)
        self.grid_columnconfigure(2, weight=1, minsize=2)
        self.master.protocol('WM_DELETE_WINDOW', sys.exit)

    def getIdentityFile(self):
        r = tkFileDialog.askopenfilename()
        if r:
            self.identity.delete(0, Tkinter.END)
            self.identity.insert(Tkinter.END, r)

    def addForward(self):
        port = self.forwardPort.get()
        self.forwardPort.delete(0, Tkinter.END)
        host = self.forwardHost.get()
        self.forwardHost.delete(0, Tkinter.END)
        if self.localRemoteVar.get() == 'local':
            self.forwards.insert(Tkinter.END, f'L:{port}:{host}')
        else:
            self.forwards.insert(Tkinter.END, f'R:{port}:{host}')

    def removeForward(self):
        cur = self.forwards.curselection()
        if cur:
            self.forwards.remove(cur[0])

    def doConnect(self):
        finished = 1
        options['host'] = self.host.get()
        options['port'] = self.port.get()
        options['user'] = self.user.get()
        options['command'] = self.command.get()
        cipher = self.cipher.get()
        mac = self.mac.get()
        escape = self.escape.get()
        if cipher:
            if cipher in SSHClientTransport.supportedCiphers:
                SSHClientTransport.supportedCiphers = [cipher]
            else:
                tkMessageBox.showerror('TkConch', 'Bad cipher.')
                finished = 0
        if mac:
            if mac in SSHClientTransport.supportedMACs:
                SSHClientTransport.supportedMACs = [mac]
            elif finished:
                tkMessageBox.showerror('TkConch', 'Bad MAC.')
                finished = 0
        if escape:
            if escape == 'none':
                options['escape'] = None
            elif escape[0] == '^' and len(escape) == 2:
                options['escape'] = chr(ord(escape[1]) - 64)
            elif len(escape) == 1:
                options['escape'] = escape
            elif finished:
                tkMessageBox.showerror('TkConch', "Bad escape character '%s'." % escape)
                finished = 0
        if self.identity.get():
            options.identitys.append(self.identity.get())
        for line in self.forwards.get(0, Tkinter.END):
            if line[0] == 'L':
                options.opt_localforward(line[2:])
            else:
                options.opt_remoteforward(line[2:])
        if '@' in options['host']:
            options['user'], options['host'] = options['host'].split('@', 1)
        if (not options['host'] or not options['user']) and finished:
            tkMessageBox.showerror('TkConch', 'Missing host or username.')
            finished = 0
        if finished:
            self.master.quit()
            self.master.destroy()
            if options['log']:
                realout = sys.stdout
                log.startLogging(sys.stderr)
                sys.stdout = realout
            else:
                log.discardLogs()
            log.deferr = handleError
            if not options.identitys:
                options.identitys = ['~/.ssh/id_rsa', '~/.ssh/id_dsa']
            host = options['host']
            port = int(options['port'] or 22)
            log.msg((host, port))
            reactor.connectTCP(host, port, SSHClientFactory())
            frame.master.deiconify()
            frame.master.title('{}@{} - TkConch'.format(options['user'], options['host']))
        else:
            self.focus()