import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
def _really_handle(self):
    """handle() without the exception handling."""
    self.server.log.debug('Received: %r', self.packet)
    request = email.message_from_bytes(self.rfile.read().replace(b'\n\n', b'\n') + b'\n')
    self.response['Thread'] = request['Thread']
    user = request['User'] or pyzor.anonymous_user
    if user != pyzor.anonymous_user:
        try:
            pyzor.account.verify_signature(request, self.server.accounts[user])
        except KeyError:
            raise pyzor.SignatureError('Unknown user.')
    if 'PV' not in request:
        raise pyzor.ProtocolError('Protocol Version not specified in request')
    try:
        if int(float(request['PV'])) != int(pyzor.proto_version):
            raise pyzor.UnsupportedVersionError()
    except ValueError:
        self.server.log.warn('Invalid PV: %s', request['PV'])
        raise pyzor.ProtocolError('Invalid Protocol Version')
    opcode = request['Op']
    if opcode not in self.server.acl[user]:
        raise pyzor.AuthorizationError('User is not authorized to request the operation.')
    self.server.log.debug('Got a %s command from %s', opcode, self.client_address[0])
    try:
        dispatch = self.dispatches[opcode]
    except KeyError:
        raise NotImplementedError('Requested operation is not implemented.')
    digests = request.get_all('Op-Digest')
    if dispatch and digests:
        dispatch(self, digests)
    self.server.usage_log.info('%s,%s,%s,%r,%s', user, self.client_address[0], opcode, digests, self.response['Code'])